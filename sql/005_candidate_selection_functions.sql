-- =====================================================
-- Candidate Selection Functions for Recommendation Pipeline
-- Run this after previous migrations
--
-- Functions:
-- 1. match_products_with_hard_filters - pgvector search with onboarding filters
-- 2. get_trending_with_hard_filters - trending fallback with filters
-- 3. save_onboarding_profile - store 9-module onboarding data
-- 4. get_user_state - retrieve user's recommendation state
-- =====================================================

-- =====================================================
-- STEP 0: Ensure products table has required columns
-- Add these columns if they don't exist
-- =====================================================

-- Add attribute columns to products if missing
ALTER TABLE products ADD COLUMN IF NOT EXISTS colors text[] DEFAULT '{}';
ALTER TABLE products ADD COLUMN IF NOT EXISTS materials text[] DEFAULT '{}';
ALTER TABLE products ADD COLUMN IF NOT EXISTS fit text;
ALTER TABLE products ADD COLUMN IF NOT EXISTS length text;
ALTER TABLE products ADD COLUMN IF NOT EXISTS sleeve text;
ALTER TABLE products ADD COLUMN IF NOT EXISTS neckline text;
ALTER TABLE products ADD COLUMN IF NOT EXISTS style_tags text[] DEFAULT '{}';
ALTER TABLE products ADD COLUMN IF NOT EXISTS broad_category text;

-- Create indexes for filter columns
CREATE INDEX IF NOT EXISTS idx_products_colors ON products USING GIN (colors);
CREATE INDEX IF NOT EXISTS idx_products_materials ON products USING GIN (materials);
CREATE INDEX IF NOT EXISTS idx_products_broad_category ON products (broad_category);
CREATE INDEX IF NOT EXISTS idx_products_brand ON products (brand);
CREATE INDEX IF NOT EXISTS idx_products_price ON products (price);

-- =====================================================
-- STEP 1: Create user_onboarding_profiles table
-- Stores 9-module onboarding preferences
-- =====================================================
CREATE TABLE IF NOT EXISTS user_onboarding_profiles (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- User identification (one required)
    user_id         UUID UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
    anon_id         VARCHAR(64) UNIQUE,

    -- Module 1: Core Setup (HARD FILTERS)
    categories      TEXT[] DEFAULT '{}',        -- tops, bottoms, dresses, etc.
    sizes           TEXT[] DEFAULT '{}',        -- XS, S, M, L, XL
    colors_to_avoid TEXT[] DEFAULT '{}',        -- red, orange, etc.
    materials_to_avoid TEXT[] DEFAULT '{}',     -- polyester, wool, etc.

    -- Modules 2-7: Per-category preferences (JSONB for flexibility)
    tops_prefs      JSONB DEFAULT '{}',
    bottoms_prefs   JSONB DEFAULT '{}',
    skirts_prefs    JSONB DEFAULT '{}',
    dresses_prefs   JSONB DEFAULT '{}',
    one_piece_prefs JSONB DEFAULT '{}',
    outerwear_prefs JSONB DEFAULT '{}',

    -- Module 8: Style preferences
    style_directions TEXT[] DEFAULT '{}',       -- minimal, classic, trendy, statement
    modesty         VARCHAR(16),                -- modest, balanced, revealing

    -- Module 9: Brand preferences
    preferred_brands TEXT[] DEFAULT '{}',       -- boost these (soft)
    brands_to_avoid TEXT[] DEFAULT '{}',        -- exclude these (hard)
    brand_openness  VARCHAR(32),                -- stick_to_favorites, mix, discover_new

    -- Global price range (optional override)
    global_min_price NUMERIC,
    global_max_price NUMERIC,

    -- Timestamps
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT onboarding_has_identity CHECK (user_id IS NOT NULL OR anon_id IS NOT NULL)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_onboarding_user ON user_onboarding_profiles (user_id) WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_onboarding_anon ON user_onboarding_profiles (anon_id) WHERE anon_id IS NOT NULL;

-- =====================================================
-- STEP 2: match_products_with_hard_filters
-- pgvector search with hard filters from onboarding
-- =====================================================
CREATE OR REPLACE FUNCTION match_products_with_hard_filters(
    query_embedding vector(512),
    filter_gender text DEFAULT NULL,
    filter_categories text[] DEFAULT NULL,      -- HARD: from Module 1
    exclude_colors text[] DEFAULT NULL,         -- HARD: from Module 1
    exclude_materials text[] DEFAULT NULL,      -- HARD: from Module 1
    exclude_brands text[] DEFAULT NULL,         -- HARD: from Module 9
    min_price numeric DEFAULT NULL,
    max_price numeric DEFAULT NULL,
    exclude_product_ids text[] DEFAULT NULL,    -- disliked + session_seen
    match_count int DEFAULT 500
)
RETURNS TABLE(
    product_id uuid,
    name text,
    brand text,
    category text,
    broad_category text,
    colors text[],
    materials text[],
    price numeric,
    fit text,
    length text,
    sleeve text,
    neckline text,
    style_tags text[],
    primary_image_url text,
    hero_image_url text,
    similarity float
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id as product_id,
        p.name,
        p.brand,
        p.category,
        p.broad_category,
        p.colors,
        p.materials,
        p.price,
        p.fit,
        p.length,
        p.sleeve,
        p.neckline,
        p.style_tags,
        p.primary_image_url,
        p.hero_image_url,
        1 - (ie.embedding <=> query_embedding) as similarity
    FROM products p
    JOIN image_embeddings ie ON ie.sku_id = p.id
    WHERE p.in_stock = true
      -- Gender filter
      AND (filter_gender IS NULL OR filter_gender = ANY(p.gender))
      -- Category filter (broad_category or category)
      AND (filter_categories IS NULL
           OR p.broad_category = ANY(filter_categories)
           OR p.category = ANY(filter_categories))
      -- Price range
      AND (min_price IS NULL OR p.price >= min_price)
      AND (max_price IS NULL OR p.price <= max_price)
      -- Exclude colors (array overlap check)
      AND (exclude_colors IS NULL OR NOT (p.colors && exclude_colors))
      -- Exclude materials
      AND (exclude_materials IS NULL OR NOT (p.materials && exclude_materials))
      -- Exclude brands
      AND (exclude_brands IS NULL OR p.brand != ALL(exclude_brands))
      -- Exclude specific products (disliked + seen)
      AND (exclude_product_ids IS NULL OR p.id::text != ALL(exclude_product_ids))
      -- Ensure embedding exists
      AND ie.embedding IS NOT NULL
    ORDER BY ie.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- =====================================================
-- STEP 3: get_trending_with_hard_filters
-- Trending products with same hard filters as above
-- =====================================================
CREATE OR REPLACE FUNCTION get_trending_with_hard_filters(
    filter_gender text DEFAULT NULL,
    filter_categories text[] DEFAULT NULL,
    exclude_colors text[] DEFAULT NULL,
    exclude_materials text[] DEFAULT NULL,
    exclude_brands text[] DEFAULT NULL,
    min_price numeric DEFAULT NULL,
    max_price numeric DEFAULT NULL,
    exclude_product_ids text[] DEFAULT NULL,
    result_limit int DEFAULT 100
)
RETURNS TABLE(
    product_id uuid,
    name text,
    brand text,
    category text,
    broad_category text,
    colors text[],
    materials text[],
    price numeric,
    fit text,
    length text,
    sleeve text,
    neckline text,
    style_tags text[],
    primary_image_url text,
    hero_image_url text,
    trending_score float,
    -- embedding_score is 0 for trending (no taste_vector comparison)
    similarity float
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id as product_id,
        p.name,
        p.brand,
        p.category,
        p.broad_category,
        p.colors,
        p.materials,
        p.price,
        p.fit,
        p.length,
        p.sleeve,
        p.neckline,
        p.style_tags,
        p.primary_image_url,
        p.hero_image_url,
        COALESCE(p.trending_score, 0.0)::float as trending_score,
        0.0::float as similarity  -- No embedding comparison for trending
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
      AND (exclude_product_ids IS NULL OR p.id::text != ALL(exclude_product_ids))
    ORDER BY COALESCE(p.trending_score, 0) DESC, p.created_at DESC
    LIMIT result_limit;
END;
$$;

-- =====================================================
-- STEP 4: get_random_exploration_items
-- Random diverse items for discovery (10% exploration)
-- =====================================================
CREATE OR REPLACE FUNCTION get_random_exploration_items(
    filter_gender text DEFAULT NULL,
    filter_categories text[] DEFAULT NULL,
    exclude_colors text[] DEFAULT NULL,
    exclude_materials text[] DEFAULT NULL,
    exclude_brands text[] DEFAULT NULL,
    exclude_product_ids text[] DEFAULT NULL,
    result_limit int DEFAULT 50
)
RETURNS TABLE(
    product_id uuid,
    name text,
    brand text,
    category text,
    broad_category text,
    colors text[],
    materials text[],
    price numeric,
    fit text,
    primary_image_url text,
    hero_image_url text
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id as product_id,
        p.name,
        p.brand,
        p.category,
        p.broad_category,
        p.colors,
        p.materials,
        p.price,
        p.fit,
        p.primary_image_url,
        p.hero_image_url
    FROM products p
    WHERE p.in_stock = true
      AND (filter_gender IS NULL OR filter_gender = ANY(p.gender))
      AND (filter_categories IS NULL
           OR p.broad_category = ANY(filter_categories)
           OR p.category = ANY(filter_categories))
      AND (exclude_colors IS NULL OR NOT (p.colors && exclude_colors))
      AND (exclude_materials IS NULL OR NOT (p.materials && exclude_materials))
      AND (exclude_brands IS NULL OR p.brand != ALL(exclude_brands))
      AND (exclude_product_ids IS NULL OR p.id::text != ALL(exclude_product_ids))
    ORDER BY RANDOM()
    LIMIT result_limit;
END;
$$;

-- =====================================================
-- STEP 5: save_onboarding_profile
-- Upserts user's 9-module onboarding profile
-- =====================================================
CREATE OR REPLACE FUNCTION save_onboarding_profile(
    p_user_id uuid DEFAULT NULL,
    p_anon_id varchar DEFAULT NULL,
    -- Module 1: Core Setup
    p_categories text[] DEFAULT '{}',
    p_sizes text[] DEFAULT '{}',
    p_colors_to_avoid text[] DEFAULT '{}',
    p_materials_to_avoid text[] DEFAULT '{}',
    -- Modules 2-7: Per-category (as JSONB)
    p_tops_prefs jsonb DEFAULT '{}',
    p_bottoms_prefs jsonb DEFAULT '{}',
    p_skirts_prefs jsonb DEFAULT '{}',
    p_dresses_prefs jsonb DEFAULT '{}',
    p_one_piece_prefs jsonb DEFAULT '{}',
    p_outerwear_prefs jsonb DEFAULT '{}',
    -- Module 8: Style
    p_style_directions text[] DEFAULT '{}',
    p_modesty varchar DEFAULT NULL,
    -- Module 9: Brands
    p_preferred_brands text[] DEFAULT '{}',
    p_brands_to_avoid text[] DEFAULT '{}',
    p_brand_openness varchar DEFAULT NULL,
    -- Global price
    p_global_min_price numeric DEFAULT NULL,
    p_global_max_price numeric DEFAULT NULL
)
RETURNS uuid
LANGUAGE plpgsql AS $$
DECLARE
    result_id uuid;
BEGIN
    IF p_user_id IS NOT NULL THEN
        INSERT INTO user_onboarding_profiles (
            user_id,
            categories, sizes, colors_to_avoid, materials_to_avoid,
            tops_prefs, bottoms_prefs, skirts_prefs, dresses_prefs, one_piece_prefs, outerwear_prefs,
            style_directions, modesty,
            preferred_brands, brands_to_avoid, brand_openness,
            global_min_price, global_max_price,
            updated_at
        )
        VALUES (
            p_user_id,
            p_categories, p_sizes, p_colors_to_avoid, p_materials_to_avoid,
            p_tops_prefs, p_bottoms_prefs, p_skirts_prefs, p_dresses_prefs, p_one_piece_prefs, p_outerwear_prefs,
            p_style_directions, p_modesty,
            p_preferred_brands, p_brands_to_avoid, p_brand_openness,
            p_global_min_price, p_global_max_price,
            NOW()
        )
        ON CONFLICT (user_id) DO UPDATE SET
            categories = EXCLUDED.categories,
            sizes = EXCLUDED.sizes,
            colors_to_avoid = EXCLUDED.colors_to_avoid,
            materials_to_avoid = EXCLUDED.materials_to_avoid,
            tops_prefs = EXCLUDED.tops_prefs,
            bottoms_prefs = EXCLUDED.bottoms_prefs,
            skirts_prefs = EXCLUDED.skirts_prefs,
            dresses_prefs = EXCLUDED.dresses_prefs,
            one_piece_prefs = EXCLUDED.one_piece_prefs,
            outerwear_prefs = EXCLUDED.outerwear_prefs,
            style_directions = EXCLUDED.style_directions,
            modesty = EXCLUDED.modesty,
            preferred_brands = EXCLUDED.preferred_brands,
            brands_to_avoid = EXCLUDED.brands_to_avoid,
            brand_openness = EXCLUDED.brand_openness,
            global_min_price = EXCLUDED.global_min_price,
            global_max_price = EXCLUDED.global_max_price,
            updated_at = NOW()
        RETURNING id INTO result_id;
    ELSIF p_anon_id IS NOT NULL THEN
        INSERT INTO user_onboarding_profiles (
            anon_id,
            categories, sizes, colors_to_avoid, materials_to_avoid,
            tops_prefs, bottoms_prefs, skirts_prefs, dresses_prefs, one_piece_prefs, outerwear_prefs,
            style_directions, modesty,
            preferred_brands, brands_to_avoid, brand_openness,
            global_min_price, global_max_price,
            updated_at
        )
        VALUES (
            p_anon_id,
            p_categories, p_sizes, p_colors_to_avoid, p_materials_to_avoid,
            p_tops_prefs, p_bottoms_prefs, p_skirts_prefs, p_dresses_prefs, p_one_piece_prefs, p_outerwear_prefs,
            p_style_directions, p_modesty,
            p_preferred_brands, p_brands_to_avoid, p_brand_openness,
            p_global_min_price, p_global_max_price,
            NOW()
        )
        ON CONFLICT (anon_id) DO UPDATE SET
            categories = EXCLUDED.categories,
            sizes = EXCLUDED.sizes,
            colors_to_avoid = EXCLUDED.colors_to_avoid,
            materials_to_avoid = EXCLUDED.materials_to_avoid,
            tops_prefs = EXCLUDED.tops_prefs,
            bottoms_prefs = EXCLUDED.bottoms_prefs,
            skirts_prefs = EXCLUDED.skirts_prefs,
            dresses_prefs = EXCLUDED.dresses_prefs,
            one_piece_prefs = EXCLUDED.one_piece_prefs,
            outerwear_prefs = EXCLUDED.outerwear_prefs,
            style_directions = EXCLUDED.style_directions,
            modesty = EXCLUDED.modesty,
            preferred_brands = EXCLUDED.preferred_brands,
            brands_to_avoid = EXCLUDED.brands_to_avoid,
            brand_openness = EXCLUDED.brand_openness,
            global_min_price = EXCLUDED.global_min_price,
            global_max_price = EXCLUDED.global_max_price,
            updated_at = NOW()
        RETURNING id INTO result_id;
    ELSE
        RAISE EXCEPTION 'Either user_id or anon_id must be provided';
    END IF;

    RETURN result_id;
END;
$$;

-- =====================================================
-- STEP 6: get_user_recommendation_state
-- Get combined user state for recommendation pipeline
-- =====================================================
CREATE OR REPLACE FUNCTION get_user_recommendation_state(
    p_user_id uuid DEFAULT NULL,
    p_anon_id varchar DEFAULT NULL
)
RETURNS TABLE(
    -- User info
    user_identifier text,
    -- From Tinder test
    has_taste_vector boolean,
    taste_vector vector(512),
    tinder_categories_tested text[],
    attribute_preferences jsonb,
    -- From Onboarding
    has_onboarding boolean,
    onboarding_categories text[],
    colors_to_avoid text[],
    materials_to_avoid text[],
    brands_to_avoid text[],
    preferred_brands text[],
    style_directions text[],
    -- User state type
    state_type text
)
LANGUAGE plpgsql AS $$
DECLARE
    v_has_taste boolean;
    v_taste vector(512);
    v_tinder_cats text[];
    v_attr_prefs jsonb;
    v_has_onboard boolean;
    v_onboard_cats text[];
    v_colors text[];
    v_materials text[];
    v_brands_avoid text[];
    v_brands_prefer text[];
    v_styles text[];
    v_state text;
BEGIN
    -- Get Tinder test data
    SELECT
        sp.taste_vector IS NOT NULL,
        sp.taste_vector,
        sp.categories_tested,
        sp.attribute_preferences
    INTO v_has_taste, v_taste, v_tinder_cats, v_attr_prefs
    FROM user_seed_preferences sp
    WHERE (p_user_id IS NOT NULL AND sp.user_id = p_user_id)
       OR (p_anon_id IS NOT NULL AND sp.anon_id = p_anon_id)
    LIMIT 1;

    -- Get Onboarding data
    SELECT
        op.id IS NOT NULL,
        op.categories,
        op.colors_to_avoid,
        op.materials_to_avoid,
        op.brands_to_avoid,
        op.preferred_brands,
        op.style_directions
    INTO v_has_onboard, v_onboard_cats, v_colors, v_materials, v_brands_avoid, v_brands_prefer, v_styles
    FROM user_onboarding_profiles op
    WHERE (p_user_id IS NOT NULL AND op.user_id = p_user_id)
       OR (p_anon_id IS NOT NULL AND op.anon_id = p_anon_id)
    LIMIT 1;

    -- Determine state type
    -- (Would check interaction_sequence count in Python, not SQL)
    IF v_has_taste THEN
        v_state := 'tinder_complete';
    ELSE
        v_state := 'cold_start';
    END IF;

    RETURN QUERY SELECT
        COALESCE(p_user_id::text, p_anon_id),
        COALESCE(v_has_taste, false),
        v_taste,
        COALESCE(v_tinder_cats, '{}'),
        COALESCE(v_attr_prefs, '{}'),
        COALESCE(v_has_onboard, false),
        COALESCE(v_onboard_cats, '{}'),
        COALESCE(v_colors, '{}'),
        COALESCE(v_materials, '{}'),
        COALESCE(v_brands_avoid, '{}'),
        COALESCE(v_brands_prefer, '{}'),
        COALESCE(v_styles, '{}'),
        v_state;
END;
$$;

-- =====================================================
-- STEP 7: Apply updated_at trigger to new table
-- =====================================================
DROP TRIGGER IF EXISTS update_user_onboarding_profiles_updated_at ON user_onboarding_profiles;
CREATE TRIGGER update_user_onboarding_profiles_updated_at
    BEFORE UPDATE ON user_onboarding_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- Done! Test the functions:
-- =====================================================
-- SELECT * FROM match_products_with_hard_filters(
--     (SELECT taste_vector FROM user_seed_preferences WHERE anon_id = 'test_user' LIMIT 1),
--     'female',
--     ARRAY['tops', 'dresses'],
--     ARRAY['red', 'orange'],
--     ARRAY['polyester'],
--     NULL,
--     NULL,
--     100.0,
--     NULL,
--     300
-- );
