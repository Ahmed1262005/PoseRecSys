-- =====================================================
-- Onboarding V2 - 10-Module Support
-- Run this after 005_candidate_selection_functions.sql
--
-- Changes:
-- 1. Add new columns to user_onboarding_profiles (birthdate, gender, style_discovery, taste_vector)
-- 2. Create save_onboarding_profile_v2 function with all 10 modules
-- =====================================================

-- =====================================================
-- STEP 1: Add new columns to user_onboarding_profiles
-- =====================================================

-- Add gender column
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS gender VARCHAR(10) DEFAULT 'female';

-- Add birthdate column
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS birthdate VARCHAR(10);

-- Add style_discovery JSONB (Module 10: Tinder test selections and summary)
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS style_discovery JSONB DEFAULT '{}';

-- Add taste_vector column (512-dim FashionCLIP embedding from style discovery)
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS taste_vector vector(512);

-- Add completed_at timestamp
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS completed_at TIMESTAMPTZ;

-- Create index on taste_vector for similarity search
CREATE INDEX IF NOT EXISTS idx_onboarding_taste_vector
ON user_onboarding_profiles
USING ivfflat (taste_vector vector_cosine_ops)
WHERE taste_vector IS NOT NULL;

-- =====================================================
-- STEP 2: save_onboarding_profile_v2
-- Upserts user's 10-module onboarding profile
-- =====================================================
CREATE OR REPLACE FUNCTION save_onboarding_profile_v2(
    p_user_id uuid DEFAULT NULL,
    p_anon_id varchar DEFAULT NULL,
    p_gender varchar DEFAULT 'female',
    -- Module 1: Core Setup
    p_categories text[] DEFAULT '{}',
    p_sizes text[] DEFAULT '{}',
    p_birthdate varchar DEFAULT NULL,
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
    p_global_max_price numeric DEFAULT NULL,
    -- Module 10: Style Discovery
    p_style_discovery jsonb DEFAULT NULL,
    p_taste_vector float8[] DEFAULT NULL,
    -- Metadata
    p_completed_at varchar DEFAULT NULL
)
RETURNS uuid
LANGUAGE plpgsql AS $$
DECLARE
    result_id uuid;
    v_taste_vector vector(512);
    v_completed_at timestamptz;
BEGIN
    -- Convert taste_vector array to vector type
    IF p_taste_vector IS NOT NULL AND array_length(p_taste_vector, 1) = 512 THEN
        v_taste_vector := p_taste_vector::vector(512);
    ELSE
        v_taste_vector := NULL;
    END IF;

    -- Parse completed_at timestamp
    IF p_completed_at IS NOT NULL THEN
        BEGIN
            v_completed_at := p_completed_at::timestamptz;
        EXCEPTION WHEN OTHERS THEN
            v_completed_at := NOW();
        END;
    ELSE
        v_completed_at := NOW();
    END IF;

    IF p_user_id IS NOT NULL THEN
        INSERT INTO user_onboarding_profiles (
            user_id,
            gender,
            categories, sizes, birthdate, colors_to_avoid, materials_to_avoid,
            tops_prefs, bottoms_prefs, skirts_prefs, dresses_prefs, one_piece_prefs, outerwear_prefs,
            style_directions, modesty,
            preferred_brands, brands_to_avoid, brand_openness,
            global_min_price, global_max_price,
            style_discovery, taste_vector,
            completed_at, updated_at
        )
        VALUES (
            p_user_id,
            p_gender,
            p_categories, p_sizes, p_birthdate, p_colors_to_avoid, p_materials_to_avoid,
            p_tops_prefs, p_bottoms_prefs, p_skirts_prefs, p_dresses_prefs, p_one_piece_prefs, p_outerwear_prefs,
            p_style_directions, p_modesty,
            p_preferred_brands, p_brands_to_avoid, p_brand_openness,
            p_global_min_price, p_global_max_price,
            COALESCE(p_style_discovery, '{}'), v_taste_vector,
            v_completed_at, NOW()
        )
        ON CONFLICT (user_id) DO UPDATE SET
            gender = EXCLUDED.gender,
            categories = EXCLUDED.categories,
            sizes = EXCLUDED.sizes,
            birthdate = EXCLUDED.birthdate,
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
            style_discovery = EXCLUDED.style_discovery,
            taste_vector = EXCLUDED.taste_vector,
            completed_at = EXCLUDED.completed_at,
            updated_at = NOW()
        RETURNING id INTO result_id;
    ELSIF p_anon_id IS NOT NULL THEN
        INSERT INTO user_onboarding_profiles (
            anon_id,
            gender,
            categories, sizes, birthdate, colors_to_avoid, materials_to_avoid,
            tops_prefs, bottoms_prefs, skirts_prefs, dresses_prefs, one_piece_prefs, outerwear_prefs,
            style_directions, modesty,
            preferred_brands, brands_to_avoid, brand_openness,
            global_min_price, global_max_price,
            style_discovery, taste_vector,
            completed_at, updated_at
        )
        VALUES (
            p_anon_id,
            p_gender,
            p_categories, p_sizes, p_birthdate, p_colors_to_avoid, p_materials_to_avoid,
            p_tops_prefs, p_bottoms_prefs, p_skirts_prefs, p_dresses_prefs, p_one_piece_prefs, p_outerwear_prefs,
            p_style_directions, p_modesty,
            p_preferred_brands, p_brands_to_avoid, p_brand_openness,
            p_global_min_price, p_global_max_price,
            COALESCE(p_style_discovery, '{}'), v_taste_vector,
            v_completed_at, NOW()
        )
        ON CONFLICT (anon_id) DO UPDATE SET
            gender = EXCLUDED.gender,
            categories = EXCLUDED.categories,
            sizes = EXCLUDED.sizes,
            birthdate = EXCLUDED.birthdate,
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
            style_discovery = EXCLUDED.style_discovery,
            taste_vector = EXCLUDED.taste_vector,
            completed_at = EXCLUDED.completed_at,
            updated_at = NOW()
        RETURNING id INTO result_id;
    ELSE
        RAISE EXCEPTION 'Either user_id or anon_id must be provided';
    END IF;

    RETURN result_id;
END;
$$;

-- =====================================================
-- STEP 3: Update get_user_recommendation_state
-- Include taste_vector from onboarding profile
-- =====================================================
CREATE OR REPLACE FUNCTION get_user_recommendation_state(
    p_user_id uuid DEFAULT NULL,
    p_anon_id varchar DEFAULT NULL
)
RETURNS TABLE(
    -- User info
    user_identifier text,
    -- From Tinder test (legacy) or onboarding style_discovery
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
    v_has_taste boolean := false;
    v_taste vector(512);
    v_tinder_cats text[];
    v_attr_prefs jsonb;
    v_has_onboard boolean := false;
    v_onboard_cats text[];
    v_colors text[];
    v_materials text[];
    v_brands_avoid text[];
    v_brands_prefer text[];
    v_styles text[];
    v_state text;
    v_style_discovery jsonb;
BEGIN
    -- First try to get from onboarding profile (new source)
    SELECT
        op.id IS NOT NULL,
        op.taste_vector,
        op.categories,
        op.colors_to_avoid,
        op.materials_to_avoid,
        op.brands_to_avoid,
        op.preferred_brands,
        op.style_directions,
        op.style_discovery
    INTO v_has_onboard, v_taste, v_onboard_cats, v_colors, v_materials, v_brands_avoid, v_brands_prefer, v_styles, v_style_discovery
    FROM user_onboarding_profiles op
    WHERE (p_user_id IS NOT NULL AND op.user_id = p_user_id)
       OR (p_anon_id IS NOT NULL AND op.anon_id = p_anon_id)
    LIMIT 1;

    -- Check if taste vector exists in onboarding
    IF v_taste IS NOT NULL THEN
        v_has_taste := true;
        -- Extract categories tested from style_discovery if available
        IF v_style_discovery IS NOT NULL AND v_style_discovery ? 'selections' THEN
            SELECT ARRAY_AGG(DISTINCT s->>'category')
            INTO v_tinder_cats
            FROM jsonb_array_elements(v_style_discovery->'selections') s
            WHERE s->>'category' IS NOT NULL;
        END IF;
        -- Extract attribute preferences from style_discovery summary
        IF v_style_discovery IS NOT NULL AND v_style_discovery ? 'summary' THEN
            v_attr_prefs := v_style_discovery->'summary'->'attribute_preferences';
        END IF;
    ELSE
        -- Fallback: Check legacy user_seed_preferences table
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
    END IF;

    -- Determine state type
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
-- Done!
--
-- To apply these changes:
-- 1. Run this SQL in Supabase Dashboard SQL Editor
-- 2. Verify with: SELECT * FROM user_onboarding_profiles LIMIT 1;
-- =====================================================
