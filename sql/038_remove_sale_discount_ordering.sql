-- =====================================================
-- Remove strict discount ordering from sale items
--
-- Problem: Sale items were ordered strictly by discount %,
-- which prevented user's preferred brands from appearing
-- if they had lower discounts.
--
-- Solution: Use standard exploration ordering (randomized)
-- for sale items. Python soft scoring handles both:
-- - Discount weighting (via preference_score)
-- - Brand preferences (+0.80 boost for preferred brands)
-- =====================================================

-- Update get_exploration_keyset - remove discount-based ordering
DROP FUNCTION IF EXISTS get_exploration_keyset(text, text[], text[], text[], text[], numeric, numeric, text, float, uuid, int, text[], text[], text[], float, float, text[], text[], float, boolean, boolean, int);

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
    -- Sale and New Arrivals filters
    on_sale_only boolean DEFAULT false,
    new_arrivals_only boolean DEFAULT false,
    new_arrivals_days int DEFAULT 7
)
RETURNS TABLE(
    product_id uuid,
    name text,
    brand text,
    category text,
    broad_category text,
    article_type text,
    colors text[],
    materials text[],
    price numeric,
    fit text,
    length text,
    sleeve text,
    neckline text,
    rise text,
    style_tags text[],
    primary_image_url text,
    hero_image_url text,
    gallery_images text[],
    exploration_score float,
    similarity float,
    computed_occasion_scores jsonb,
    computed_style_scores jsonb,
    computed_pattern_scores jsonb,
    -- Sale/New return fields
    original_price numeric,
    discount_percent float,
    is_on_sale boolean,
    is_new boolean,
    created_at timestamptz
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
        p.article_type,
        p.colors,
        p.materials,
        p.price,
        p.fit,
        p.length,
        p.sleeve,
        p.neckline,
        p.rise,
        p.style_tags,
        p.primary_image_url,
        p.hero_image_url,
        p.gallery_images,
        -- Score calculation: Use random exploration score for ALL cases
        -- Python soft scoring handles discount weighting + brand preferences
        CASE
            WHEN new_arrivals_only THEN
                -- For new arrivals, order by recency (newer = higher score)
                EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')
            ELSE
                -- Standard exploration: random but deterministic ordering
                -- Used for both regular feed AND sale items
                (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
        END as exploration_score,
        -- Similarity (same as exploration_score for consistency)
        CASE
            WHEN new_arrivals_only THEN
                EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')
            ELSE
                (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
        END as similarity,
        COALESCE(p.computed_occasion_scores, '{}'::jsonb) as computed_occasion_scores,
        COALESCE(p.computed_style_scores, '{}'::jsonb) as computed_style_scores,
        COALESCE(p.computed_pattern_scores, '{}'::jsonb) as computed_pattern_scores,
        -- Sale/New fields (still returned for display, just not used for ordering)
        p.original_price,
        CASE
            WHEN p.original_price IS NOT NULL AND p.original_price > 0 AND p.price > 0 AND p.original_price > p.price
            THEN ROUND((1 - p.price / p.original_price) * 100)::float
            ELSE NULL
        END as discount_percent,
        CASE
            WHEN p.original_price IS NOT NULL AND p.original_price > p.price
            THEN true
            ELSE false
        END as is_on_sale,
        CASE
            WHEN p.created_at >= NOW() - (new_arrivals_days || ' days')::interval
            THEN true
            ELSE false
        END as is_new,
        p.created_at
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
      AND NOT has_excluded_style(p.computed_style_scores, exclude_styles, style_threshold)
      AND matches_occasion(p.computed_occasion_scores, include_occasions, occasion_threshold)
      AND matches_article_type(p.article_type, include_article_types)
      AND matches_pattern(p.computed_pattern_scores, include_patterns, pattern_threshold)
      AND NOT has_excluded_pattern(p.computed_pattern_scores, exclude_patterns, pattern_threshold)
      -- Sale filter (still filters to only sale items, just doesn't order by discount)
      AND (NOT on_sale_only OR (p.original_price IS NOT NULL AND p.original_price > p.price AND p.price > 0))
      -- New arrivals filter
      AND (NOT new_arrivals_only OR p.created_at >= NOW() - (new_arrivals_days || ' days')::interval)
      -- Keyset cursor condition
      AND (
          cursor_score IS NULL
          OR (
              CASE
                  WHEN new_arrivals_only THEN
                      EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')
                  ELSE
                      (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
              END
          ) < cursor_score
          OR (
              (
                  CASE
                      WHEN new_arrivals_only THEN
                          EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')
                      ELSE
                          (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
                  END
              ) = cursor_score
              AND p.id < cursor_id
          )
      )
    ORDER BY (
        CASE
            WHEN new_arrivals_only THEN
                EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')
            ELSE
                (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
        END
    ) DESC, p.id DESC
    LIMIT p_limit;
END;
$$;

GRANT EXECUTE ON FUNCTION get_exploration_keyset TO anon, authenticated;


-- Update get_exploration_keyset_with_brands - remove discount-based ordering
DROP FUNCTION IF EXISTS get_exploration_keyset_with_brands(text, text[], text[], text[], text[], numeric, numeric, text, float, uuid, int, text[], text[], text[], float, float, text[], text[], float, text[], boolean, boolean, int);

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
    -- Sale and New Arrivals filters
    on_sale_only boolean DEFAULT false,
    new_arrivals_only boolean DEFAULT false,
    new_arrivals_days int DEFAULT 7
)
RETURNS TABLE(
    product_id uuid,
    name text,
    brand text,
    category text,
    broad_category text,
    article_type text,
    colors text[],
    materials text[],
    price numeric,
    fit text,
    length text,
    sleeve text,
    neckline text,
    rise text,
    style_tags text[],
    primary_image_url text,
    hero_image_url text,
    gallery_images text[],
    exploration_score float,
    similarity float,
    is_preferred_brand boolean,
    computed_occasion_scores jsonb,
    computed_style_scores jsonb,
    computed_pattern_scores jsonb,
    -- Sale/New return fields
    original_price numeric,
    discount_percent float,
    is_on_sale boolean,
    is_new boolean,
    created_at timestamptz
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
        p.article_type,
        p.colors,
        p.materials,
        p.price,
        p.fit,
        p.length,
        p.sleeve,
        p.neckline,
        p.rise,
        p.style_tags,
        p.primary_image_url,
        p.hero_image_url,
        p.gallery_images,
        -- Score calculation: Use random exploration score with brand boost
        -- No special discount ordering - Python handles discount weighting
        CASE
            WHEN new_arrivals_only THEN
                -- For new arrivals, order by recency with brand boost
                CASE
                    WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
                    THEN (EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')) + 0.5
                    ELSE EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')
                END
            ELSE
                -- Standard exploration with brand boost (for both regular AND sale items)
                CASE
                    WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
                    THEN (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float + 0.5
                    ELSE (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
                END
        END as exploration_score,
        -- Similarity (same as exploration_score)
        CASE
            WHEN new_arrivals_only THEN
                CASE
                    WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
                    THEN (EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')) + 0.5
                    ELSE EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')
                END
            ELSE
                CASE
                    WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
                    THEN (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float + 0.5
                    ELSE (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
                END
        END as similarity,
        CASE
            WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
            THEN true
            ELSE false
        END as is_preferred_brand,
        COALESCE(p.computed_occasion_scores, '{}'::jsonb) as computed_occasion_scores,
        COALESCE(p.computed_style_scores, '{}'::jsonb) as computed_style_scores,
        COALESCE(p.computed_pattern_scores, '{}'::jsonb) as computed_pattern_scores,
        -- Sale/New fields (still returned for display)
        p.original_price,
        CASE
            WHEN p.original_price IS NOT NULL AND p.original_price > 0 AND p.price > 0 AND p.original_price > p.price
            THEN ROUND((1 - p.price / p.original_price) * 100)::float
            ELSE NULL
        END as discount_percent,
        CASE
            WHEN p.original_price IS NOT NULL AND p.original_price > p.price
            THEN true
            ELSE false
        END as is_on_sale,
        CASE
            WHEN p.created_at >= NOW() - (new_arrivals_days || ' days')::interval
            THEN true
            ELSE false
        END as is_new,
        p.created_at
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
      AND NOT has_excluded_style(p.computed_style_scores, exclude_styles, style_threshold)
      AND matches_occasion(p.computed_occasion_scores, include_occasions, occasion_threshold)
      AND matches_article_type(p.article_type, include_article_types)
      AND matches_pattern(p.computed_pattern_scores, include_patterns, pattern_threshold)
      AND NOT has_excluded_pattern(p.computed_pattern_scores, exclude_patterns, pattern_threshold)
      -- Sale filter (filters to sale items, doesn't order by discount)
      AND (NOT on_sale_only OR (p.original_price IS NOT NULL AND p.original_price > p.price AND p.price > 0))
      -- New arrivals filter
      AND (NOT new_arrivals_only OR p.created_at >= NOW() - (new_arrivals_days || ' days')::interval)
      -- Keyset cursor condition
      AND (
          cursor_score IS NULL
          OR (
              CASE
                  WHEN new_arrivals_only THEN
                      CASE
                          WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
                          THEN (EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')) + 0.5
                          ELSE EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')
                      END
                  ELSE
                      CASE
                          WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
                          THEN (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float + 0.5
                          ELSE (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
                      END
              END
          ) < cursor_score
          OR (
              (
                  CASE
                      WHEN new_arrivals_only THEN
                          CASE
                              WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
                              THEN (EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')) + 0.5
                              ELSE EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')
                          END
                      ELSE
                          CASE
                              WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
                              THEN (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float + 0.5
                              ELSE (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
                          END
                  END
              ) = cursor_score
              AND p.id < cursor_id
          )
      )
    ORDER BY (
        CASE
            WHEN new_arrivals_only THEN
                CASE
                    WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
                    THEN (EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')) + 0.5
                    ELSE EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')
                END
            ELSE
                CASE
                    WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
                    THEN (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float + 0.5
                    ELSE (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
                END
        END
    ) DESC, p.id DESC
    LIMIT p_limit;
END;
$$;

GRANT EXECUTE ON FUNCTION get_exploration_keyset_with_brands TO anon, authenticated;

SELECT 'Removed strict discount ordering from sale items - now uses exploration ordering' as status;
NOTIFY pgrst, 'reload schema';
