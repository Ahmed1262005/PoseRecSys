-- =====================================================
-- Add include_product_ids parameter to exploration keyset functions
--
-- Problem: When attribute filters (formality, seasons, style_tags, etc.)
-- are active, the pipeline fetches random candidates then filters in Python.
-- With restrictive filters, most candidates get thrown away.
--
-- Solution: Pre-filter via product_attributes to get matching product IDs,
-- then pass as include_product_ids to the SQL RPC. Every row returned
-- already satisfies the attribute filters.
--
-- Logic: AND (include_product_ids IS NULL OR p.id = ANY(include_product_ids))
-- =====================================================

-- =============================================================
-- get_exploration_keyset: Add include_product_ids parameter
-- =============================================================

DROP FUNCTION IF EXISTS get_exploration_keyset(text, text[], text[], text[], text[], numeric, numeric, text, float, uuid, int, text[], text[], text[], float, float, text[], text[], float, boolean, boolean, int, uuid[], text[]);

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
    -- Pre-filtered product IDs (from product_attributes query)
    include_product_ids uuid[] DEFAULT NULL
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
      -- Pre-filtered product IDs: only return products in this set
      AND (include_product_ids IS NULL OR p.id = ANY(include_product_ids))
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
-- get_exploration_keyset_with_brands: Add include_product_ids parameter
-- =============================================================

DROP FUNCTION IF EXISTS get_exploration_keyset_with_brands(text, text[], text[], text[], text[], numeric, numeric, text, float, uuid, int, text[], text[], text[], float, float, text[], text[], float, text[], boolean, boolean, int, uuid[], text[]);

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
    -- Pre-filtered product IDs (from product_attributes query)
    include_product_ids uuid[] DEFAULT NULL
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
      -- Pre-filtered product IDs: only return products in this set
      AND (include_product_ids IS NULL OR p.id = ANY(include_product_ids))
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

SELECT 'Added include_product_ids parameter to exploration keyset functions for pre-filtered attribute queries' as status;
NOTIFY pgrst, 'reload schema';
