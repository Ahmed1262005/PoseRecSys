-- =============================================================================
-- V3 Feed: Slim Retrieval Functions
-- =============================================================================
--
-- Three focused RPCs replacing the monolithic get_exploration_keyset.
-- Each is simple, indexed, and cheap. No exclude arrays. No md5 ordering.
--
-- 1. v3_get_candidates_by_explore_key  — keyset pagination over explore keys
-- 2. v3_get_candidates_by_freshness    — new arrivals / freshness ordering
-- 3. v3_hydrate_candidates             — batch feature fetch by ID array
--
-- DEPENDS ON: 062_v3_product_serving_view.sql (product_serving MV)
-- RUN ORDER: 061 first, then 062, then 063.
-- =============================================================================


-- =============================================================================
-- Function 1: Candidates by explore key
-- =============================================================================
--
-- Keyset pagination over precomputed explore_key_{a,b,c}.
-- Returns lightweight stubs (ID + metadata for mixing/dedup).
-- No exclude_product_ids array. No per-row md5 hashing.
-- Expected execution: < 50ms.
-- =============================================================================

CREATE OR REPLACE FUNCTION v3_get_candidates_by_explore_key(
    p_key_family      text    DEFAULT 'a',
    p_gender          text    DEFAULT NULL,
    p_categories      text[]  DEFAULT NULL,
    p_min_price       numeric DEFAULT NULL,
    p_max_price       numeric DEFAULT NULL,
    p_exclude_brands  text[]  DEFAULT NULL,
    p_include_brands  text[]  DEFAULT NULL,
    p_on_sale_only    boolean DEFAULT false,
    p_new_arrivals    boolean DEFAULT false,
    p_new_days        int     DEFAULT 30,
    p_preferred_brands text[] DEFAULT NULL,
    p_cursor_score    float   DEFAULT NULL,
    p_cursor_id       uuid    DEFAULT NULL,
    p_limit           int     DEFAULT 200
)
RETURNS TABLE (
    id              uuid,
    brand           text,
    broad_category  text,
    article_type    text,
    price           numeric,
    explore_score   float,
    image_dedup_key text
)
LANGUAGE plpgsql STABLE
AS $$
DECLARE
    v_key_col text;
BEGIN
    -- Validate key family
    IF p_key_family NOT IN ('a', 'b', 'c') THEN
        p_key_family := 'a';
    END IF;

    -- Dynamic column selection based on key family
    -- We use a CASE inside the query rather than dynamic SQL to keep
    -- the planner happy with the index.
    RETURN QUERY
    SELECT
        p.id,
        p.brand,
        p.broad_category,
        p.article_type,
        p.price,
        -- Explore score from the chosen key family
        CASE p_key_family
            WHEN 'a' THEN p.explore_key_a
            WHEN 'b' THEN p.explore_key_b
            WHEN 'c' THEN p.explore_key_c
        END AS explore_score,
        md5(COALESCE(p.primary_image_url, '')) AS image_dedup_key
    FROM products p
    WHERE p.in_stock = true
      -- Gender
      AND (p_gender IS NULL OR p_gender = ANY(p.gender))
      -- Categories
      AND (p_categories IS NULL
           OR p.broad_category = ANY(p_categories)
           OR p.category = ANY(p_categories))
      -- Price range
      AND (p_min_price IS NULL OR p.price >= p_min_price)
      AND (p_max_price IS NULL OR p.price <= p_max_price)
      -- Brand exclusion
      AND (p_exclude_brands IS NULL
           OR NOT (LOWER(p.brand) = ANY(
               SELECT LOWER(unnest(p_exclude_brands))
           )))
      -- Brand inclusion (whitelist — only these brands)
      AND (p_include_brands IS NULL
           OR LOWER(p.brand) = ANY(
               SELECT LOWER(unnest(p_include_brands))
           ))
      -- Sale only
      AND (NOT p_on_sale_only
           OR (p.original_price IS NOT NULL AND p.original_price > p.price))
      -- New arrivals
      AND (NOT p_new_arrivals
           OR p.created_at >= NOW() - (p_new_days || ' days')::interval)
      -- Keyset cursor (descending order)
      AND (
          p_cursor_score IS NULL
          OR (
              CASE p_key_family
                  WHEN 'a' THEN p.explore_key_a
                  WHEN 'b' THEN p.explore_key_b
                  WHEN 'c' THEN p.explore_key_c
              END < p_cursor_score
              OR (
                  CASE p_key_family
                      WHEN 'a' THEN p.explore_key_a
                      WHEN 'b' THEN p.explore_key_b
                      WHEN 'c' THEN p.explore_key_c
                  END = p_cursor_score
                  AND p.id < p_cursor_id
              )
          )
      )
    ORDER BY
        CASE p_key_family
            WHEN 'a' THEN p.explore_key_a
            WHEN 'b' THEN p.explore_key_b
            WHEN 'c' THEN p.explore_key_c
        END DESC,
        p.id DESC
    LIMIT p_limit;
END;
$$;


-- =============================================================================
-- Function 2: Candidates by freshness
-- =============================================================================
--
-- For new arrivals and merch source. Keyset pagination over created_at DESC.
-- Returns lightweight stubs.
-- Expected execution: < 50ms.
-- =============================================================================

CREATE OR REPLACE FUNCTION v3_get_candidates_by_freshness(
    p_gender          text        DEFAULT NULL,
    p_categories      text[]      DEFAULT NULL,
    p_min_price       numeric     DEFAULT NULL,
    p_max_price       numeric     DEFAULT NULL,
    p_exclude_brands  text[]      DEFAULT NULL,
    p_include_brands  text[]      DEFAULT NULL,
    p_on_sale_only    boolean     DEFAULT false,
    p_days            int         DEFAULT 30,
    p_cursor_date     timestamptz DEFAULT NULL,
    p_cursor_id       uuid        DEFAULT NULL,
    p_limit           int         DEFAULT 200
)
RETURNS TABLE (
    id              uuid,
    brand           text,
    broad_category  text,
    article_type    text,
    price           numeric,
    created_at      timestamptz,
    image_dedup_key text
)
LANGUAGE sql STABLE
AS $$
    SELECT
        p.id,
        p.brand,
        p.broad_category,
        p.article_type,
        p.price,
        p.created_at,
        md5(COALESCE(p.primary_image_url, '')) AS image_dedup_key
    FROM products p
    WHERE p.in_stock = true
      -- Gender
      AND (p_gender IS NULL OR p_gender = ANY(p.gender))
      -- Categories
      AND (p_categories IS NULL
           OR p.broad_category = ANY(p_categories)
           OR p.category = ANY(p_categories))
      -- Price range
      AND (p_min_price IS NULL OR p.price >= p_min_price)
      AND (p_max_price IS NULL OR p.price <= p_max_price)
      -- Brand exclusion
      AND (p_exclude_brands IS NULL
           OR NOT (LOWER(p.brand) = ANY(
               SELECT LOWER(unnest(p_exclude_brands))
           )))
      -- Brand inclusion (whitelist — only these brands)
      AND (p_include_brands IS NULL
           OR LOWER(p.brand) = ANY(
               SELECT LOWER(unnest(p_include_brands))
           ))
      -- Sale only
      AND (NOT p_on_sale_only
           OR (p.original_price IS NOT NULL AND p.original_price > p.price))
      -- Freshness window
      AND p.created_at >= NOW() - (p_days || ' days')::interval
      -- Keyset cursor (descending by created_at)
      AND (
          p_cursor_date IS NULL
          OR (p.created_at < p_cursor_date)
          OR (p.created_at = p_cursor_date AND p.id < p_cursor_id)
      )
    ORDER BY p.created_at DESC, p.id DESC
    LIMIT p_limit;
$$;


-- =============================================================================
-- Function 3: Hydrate candidates by ID array
-- =============================================================================
--
-- Batch fetch full features from the materialized view for a set of IDs.
-- Used for:
--   - Page hydration (24 items): < 20ms
--   - Pool rebuild hydration (500 items): < 100ms
--
-- Column types match product_serving MV exactly (verified via pg_attribute).
-- =============================================================================

CREATE OR REPLACE FUNCTION v3_hydrate_candidates(p_ids uuid[])
RETURNS TABLE (
    id                       uuid,
    name                     text,
    brand                    text,
    category                 text,
    broad_category           text,
    article_type             text,
    colors                   text[],
    materials                text[],
    price                    numeric(10,2),
    original_price           numeric(10,2),
    fit                      text,
    length                   text,
    sleeve                   text,
    neckline                 text,
    rise                     text,
    style_tags               text[],
    primary_image_url        text,
    hero_image_url           text,
    gallery_images           text[],
    gender                   text[],
    in_stock                 boolean,
    created_at               timestamptz,
    is_on_sale               boolean,
    discount_percent         integer,
    is_new                   boolean,
    computed_occasion_scores jsonb,
    computed_style_scores    jsonb,
    computed_pattern_scores  jsonb,
    pa_occasions             text[],
    pa_style_tags            text[],
    pa_pattern               text,
    pa_formality             text,
    pa_fit_type              text,
    pa_color_family          text,
    pa_seasons               text[],
    pa_silhouette            text,
    pa_construction          jsonb,
    pa_coverage_level        text,
    pa_skin_exposure         text,
    pa_coverage_details      jsonb,
    pa_model_body_type       text,
    pa_model_size_estimate   text,
    image_dedup_key          text,
    explore_key_a            double precision,
    explore_key_b            double precision,
    explore_key_c            double precision
)
LANGUAGE sql STABLE
AS $$
    SELECT * FROM product_serving WHERE id = ANY(p_ids);
$$;
