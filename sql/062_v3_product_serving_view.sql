-- =============================================================================
-- V3 Feed: Denormalized Product Serving View
-- =============================================================================
--
-- Materialized view joining products + product_attributes for fast
-- hydration. Eliminates the expensive LEFT JOIN on every request.
--
-- Refresh: REFRESH MATERIALIZED VIEW CONCURRENTLY product_serving;
-- Frequency: After inventory/price updates. Bump v3:catalog:version after.
--
-- Phase 1 uses MV for simplicity. If refresh cadence becomes a problem,
-- migrate to a denormalized serving table with incremental maintenance.
--
-- DEPENDS ON: 061_v3_explore_keys.sql (explore_key_a/b/c columns)
-- RUN ORDER: 061 first, then 062, then 063.
-- =============================================================================

DROP MATERIALIZED VIEW IF EXISTS product_serving;

CREATE MATERIALIZED VIEW product_serving AS
SELECT
    -- Core product fields
    p.id,
    p.name,
    p.brand,
    p.category,
    p.broad_category,
    p.article_type,
    p.colors,
    p.materials,
    p.price,
    p.original_price,
    p.fit,
    p.length,
    p.sleeve,
    p.neckline,
    p.rise,
    p.style_tags,
    p.primary_image_url,
    p.hero_image_url,
    p.gallery_images,
    p.gender,
    p.in_stock,
    p.created_at,

    -- Precomputed flags
    CASE WHEN p.original_price IS NOT NULL
          AND p.original_price > p.price
         THEN true ELSE false
    END AS is_on_sale,

    CASE WHEN p.original_price IS NOT NULL
          AND p.original_price > 0
          AND p.price > 0
          AND p.original_price > p.price
         THEN ROUND((1 - p.price / p.original_price) * 100)::int
         ELSE NULL
    END AS discount_percent,

    CASE WHEN p.created_at >= NOW() - INTERVAL '30 days'
         THEN true ELSE false
    END AS is_new,

    -- Precomputed scores (from products table)
    COALESCE(p.computed_occasion_scores, '{}'::jsonb) AS computed_occasion_scores,
    COALESCE(p.computed_style_scores, '{}'::jsonb)    AS computed_style_scores,
    COALESCE(p.computed_pattern_scores, '{}'::jsonb)  AS computed_pattern_scores,

    -- Inlined product_attributes (no join needed at hydration time)
    pa.occasions        AS pa_occasions,
    pa.style_tags       AS pa_style_tags,
    pa.pattern          AS pa_pattern,
    pa.formality        AS pa_formality,
    pa.fit_type         AS pa_fit_type,
    pa.color_family     AS pa_color_family,
    pa.seasons          AS pa_seasons,
    pa.silhouette       AS pa_silhouette,
    pa.closure_details  AS pa_construction,
    pa.coverage_level   AS pa_coverage_level,
    pa.skin_exposure    AS pa_skin_exposure,
    pa.coverage_details AS pa_coverage_details,
    pa.model_body_type  AS pa_model_body_type,
    pa.model_size_estimate AS pa_model_size_estimate,

    -- Image dedup key (hash of primary image URL)
    md5(COALESCE(p.primary_image_url, '')) AS image_dedup_key,

    -- Explore keys (populated by 061_v3_explore_keys.sql)
    p.explore_key_a,
    p.explore_key_b,
    p.explore_key_c

FROM products p
LEFT JOIN product_attributes pa ON p.id = pa.sku_id
WHERE p.in_stock = true;


-- =============================================================================
-- Indexes on the materialized view
-- =============================================================================

-- Primary lookup
CREATE UNIQUE INDEX idx_ps_id
    ON product_serving (id);

-- Category filtering
CREATE INDEX idx_ps_broad_category
    ON product_serving (broad_category);

-- Brand filtering
CREATE INDEX idx_ps_brand
    ON product_serving (brand);

-- Price range filtering + sorting
CREATE INDEX idx_ps_price
    ON product_serving (price);

-- New arrivals / freshness sorting
CREATE INDEX idx_ps_created_at
    ON product_serving (created_at DESC);

-- Sale items (partial index — only sale items)
CREATE INDEX idx_ps_on_sale
    ON product_serving (is_on_sale) WHERE is_on_sale = true;

-- New items (partial index)
CREATE INDEX idx_ps_is_new
    ON product_serving (is_new) WHERE is_new = true;

-- Gender filtering (GIN for array containment)
CREATE INDEX idx_ps_gender
    ON product_serving USING GIN (gender);

-- Color filtering (GIN for array overlap)
CREATE INDEX idx_ps_colors
    ON product_serving USING GIN (colors);

-- Keyset pagination over explore keys
CREATE INDEX idx_ps_explore_a
    ON product_serving (explore_key_a DESC, id DESC);
CREATE INDEX idx_ps_explore_b
    ON product_serving (explore_key_b DESC, id DESC);
CREATE INDEX idx_ps_explore_c
    ON product_serving (explore_key_c DESC, id DESC);


-- =============================================================================
-- Convenience function to refresh and report
-- =============================================================================

CREATE OR REPLACE FUNCTION refresh_product_serving()
RETURNS text
LANGUAGE plpgsql
AS $$
DECLARE
    row_count int;
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY product_serving;
    SELECT count(*) INTO row_count FROM product_serving;
    RETURN 'product_serving refreshed: ' || row_count || ' rows';
END;
$$;
