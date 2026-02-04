-- =====================================================
-- Onboarding V3 - New Frontend Spec Support
-- Run this after 026_update_save_onboarding_rpc.sql
--
-- This migration supports the NEW frontend onboarding format:
-- - Split sizes (topSize, bottomSize, outerwearSize)
-- - Flat attributePreferences with category mappings
-- - Simplified typePreferences
-- - stylePersona support
-- - Simplified styleDiscovery (completed + swipedItems)
--
-- CLEAN MIGRATION: Old format deprecated, only new format supported
-- =====================================================

-- =====================================================
-- STEP 1: Add new columns for split sizes
-- =====================================================
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS top_sizes text[] DEFAULT '{}';
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS bottom_sizes text[] DEFAULT '{}';
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS outerwear_sizes text[] DEFAULT '{}';

-- =====================================================
-- STEP 2: Add attribute preferences with category mappings
-- =====================================================
-- Fit preferences
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS preferred_fits text[] DEFAULT '{}';
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS fit_category_mapping jsonb DEFAULT '[]';

-- Sleeve preferences
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS preferred_sleeves text[] DEFAULT '{}';
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS sleeve_category_mapping jsonb DEFAULT '[]';

-- Length preferences (for tops/bottoms)
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS preferred_lengths text[] DEFAULT '{}';
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS length_category_mapping jsonb DEFAULT '[]';

-- Length preferences for skirts/dresses (mini, midi, maxi)
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS preferred_lengths_dresses text[] DEFAULT '{}';
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS length_dresses_category_mapping jsonb DEFAULT '[]';

-- Rise preferences
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS preferred_rises text[] DEFAULT '{}';

-- =====================================================
-- STEP 3: Add simplified type preferences
-- =====================================================
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS top_types text[] DEFAULT '{}';
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS bottom_types text[] DEFAULT '{}';
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS dress_types text[] DEFAULT '{}';
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS outerwear_types text[] DEFAULT '{}';

-- =====================================================
-- STEP 4: Add style persona and updated lifestyle fields
-- =====================================================
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS style_persona text[] DEFAULT '{}';

-- Rename pattern fields for clarity (keep old columns for backward compat during migration)
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS patterns_liked text[] DEFAULT '{}';
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS patterns_avoided text[] DEFAULT '{}';

-- =====================================================
-- STEP 5: Add simplified style discovery fields
-- =====================================================
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS style_discovery_complete boolean DEFAULT false;
ALTER TABLE user_onboarding_profiles ADD COLUMN IF NOT EXISTS swiped_items text[] DEFAULT '{}';

-- =====================================================
-- STEP 6: Create save_onboarding_profile_v3 function
-- =====================================================
DROP FUNCTION IF EXISTS save_onboarding_profile_v3 CASCADE;

CREATE OR REPLACE FUNCTION save_onboarding_profile_v3(
    p_user_id uuid DEFAULT NULL,
    p_anon_id varchar DEFAULT NULL,
    p_gender varchar DEFAULT 'female',

    -- Core Setup (with split sizes)
    p_categories text[] DEFAULT '{}',
    p_birthdate varchar DEFAULT NULL,
    p_top_sizes text[] DEFAULT '{}',
    p_bottom_sizes text[] DEFAULT '{}',
    p_outerwear_sizes text[] DEFAULT '{}',
    p_colors_to_avoid text[] DEFAULT '{}',
    p_materials_to_avoid text[] DEFAULT '{}',

    -- Attribute Preferences (flat with category mappings)
    p_preferred_fits text[] DEFAULT '{}',
    p_fit_category_mapping jsonb DEFAULT '[]',
    p_preferred_sleeves text[] DEFAULT '{}',
    p_sleeve_category_mapping jsonb DEFAULT '[]',
    p_preferred_lengths text[] DEFAULT '{}',
    p_length_category_mapping jsonb DEFAULT '[]',
    p_preferred_lengths_dresses text[] DEFAULT '{}',
    p_length_dresses_category_mapping jsonb DEFAULT '[]',
    p_preferred_rises text[] DEFAULT '{}',

    -- Type Preferences (simplified)
    p_top_types text[] DEFAULT '{}',
    p_bottom_types text[] DEFAULT '{}',
    p_dress_types text[] DEFAULT '{}',
    p_outerwear_types text[] DEFAULT '{}',

    -- Lifestyle
    p_occasions text[] DEFAULT '{}',
    p_styles_to_avoid text[] DEFAULT '{}',
    p_patterns_liked text[] DEFAULT '{}',
    p_patterns_avoided text[] DEFAULT '{}',
    p_style_persona text[] DEFAULT '{}',

    -- Brands
    p_preferred_brands text[] DEFAULT '{}',
    p_brands_to_avoid text[] DEFAULT '{}',
    p_brand_openness varchar DEFAULT NULL,

    -- Style Discovery (simplified)
    p_style_discovery_complete boolean DEFAULT false,
    p_swiped_items text[] DEFAULT '{}',
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
            user_id, gender,
            -- Core setup
            categories, birthdate, top_sizes, bottom_sizes, outerwear_sizes,
            colors_to_avoid, materials_to_avoid,
            -- Attribute preferences
            preferred_fits, fit_category_mapping,
            preferred_sleeves, sleeve_category_mapping,
            preferred_lengths, length_category_mapping,
            preferred_lengths_dresses, length_dresses_category_mapping,
            preferred_rises,
            -- Type preferences
            top_types, bottom_types, dress_types, outerwear_types,
            -- Lifestyle
            occasions, styles_to_avoid, patterns_liked, patterns_avoided, style_persona,
            -- Brands
            preferred_brands, brands_to_avoid, brand_openness,
            -- Style discovery
            style_discovery_complete, swiped_items, taste_vector,
            -- Metadata
            completed_at, updated_at
        )
        VALUES (
            p_user_id, p_gender,
            -- Core setup
            p_categories, p_birthdate, p_top_sizes, p_bottom_sizes, p_outerwear_sizes,
            p_colors_to_avoid, p_materials_to_avoid,
            -- Attribute preferences
            p_preferred_fits, p_fit_category_mapping,
            p_preferred_sleeves, p_sleeve_category_mapping,
            p_preferred_lengths, p_length_category_mapping,
            p_preferred_lengths_dresses, p_length_dresses_category_mapping,
            p_preferred_rises,
            -- Type preferences
            p_top_types, p_bottom_types, p_dress_types, p_outerwear_types,
            -- Lifestyle
            p_occasions, p_styles_to_avoid, p_patterns_liked, p_patterns_avoided, p_style_persona,
            -- Brands
            p_preferred_brands, p_brands_to_avoid, p_brand_openness,
            -- Style discovery
            p_style_discovery_complete, p_swiped_items, v_taste_vector,
            -- Metadata
            v_completed_at, NOW()
        )
        ON CONFLICT (user_id) DO UPDATE SET
            gender = EXCLUDED.gender,
            -- Core setup
            categories = EXCLUDED.categories,
            birthdate = EXCLUDED.birthdate,
            top_sizes = EXCLUDED.top_sizes,
            bottom_sizes = EXCLUDED.bottom_sizes,
            outerwear_sizes = EXCLUDED.outerwear_sizes,
            colors_to_avoid = EXCLUDED.colors_to_avoid,
            materials_to_avoid = EXCLUDED.materials_to_avoid,
            -- Attribute preferences
            preferred_fits = EXCLUDED.preferred_fits,
            fit_category_mapping = EXCLUDED.fit_category_mapping,
            preferred_sleeves = EXCLUDED.preferred_sleeves,
            sleeve_category_mapping = EXCLUDED.sleeve_category_mapping,
            preferred_lengths = EXCLUDED.preferred_lengths,
            length_category_mapping = EXCLUDED.length_category_mapping,
            preferred_lengths_dresses = EXCLUDED.preferred_lengths_dresses,
            length_dresses_category_mapping = EXCLUDED.length_dresses_category_mapping,
            preferred_rises = EXCLUDED.preferred_rises,
            -- Type preferences
            top_types = EXCLUDED.top_types,
            bottom_types = EXCLUDED.bottom_types,
            dress_types = EXCLUDED.dress_types,
            outerwear_types = EXCLUDED.outerwear_types,
            -- Lifestyle
            occasions = EXCLUDED.occasions,
            styles_to_avoid = EXCLUDED.styles_to_avoid,
            patterns_liked = EXCLUDED.patterns_liked,
            patterns_avoided = EXCLUDED.patterns_avoided,
            style_persona = EXCLUDED.style_persona,
            -- Brands
            preferred_brands = EXCLUDED.preferred_brands,
            brands_to_avoid = EXCLUDED.brands_to_avoid,
            brand_openness = EXCLUDED.brand_openness,
            -- Style discovery
            style_discovery_complete = EXCLUDED.style_discovery_complete,
            swiped_items = EXCLUDED.swiped_items,
            taste_vector = EXCLUDED.taste_vector,
            -- Metadata
            completed_at = EXCLUDED.completed_at,
            updated_at = NOW()
        RETURNING id INTO result_id;
    ELSIF p_anon_id IS NOT NULL THEN
        INSERT INTO user_onboarding_profiles (
            anon_id, gender,
            -- Core setup
            categories, birthdate, top_sizes, bottom_sizes, outerwear_sizes,
            colors_to_avoid, materials_to_avoid,
            -- Attribute preferences
            preferred_fits, fit_category_mapping,
            preferred_sleeves, sleeve_category_mapping,
            preferred_lengths, length_category_mapping,
            preferred_lengths_dresses, length_dresses_category_mapping,
            preferred_rises,
            -- Type preferences
            top_types, bottom_types, dress_types, outerwear_types,
            -- Lifestyle
            occasions, styles_to_avoid, patterns_liked, patterns_avoided, style_persona,
            -- Brands
            preferred_brands, brands_to_avoid, brand_openness,
            -- Style discovery
            style_discovery_complete, swiped_items, taste_vector,
            -- Metadata
            completed_at, updated_at
        )
        VALUES (
            p_anon_id, p_gender,
            -- Core setup
            p_categories, p_birthdate, p_top_sizes, p_bottom_sizes, p_outerwear_sizes,
            p_colors_to_avoid, p_materials_to_avoid,
            -- Attribute preferences
            p_preferred_fits, p_fit_category_mapping,
            p_preferred_sleeves, p_sleeve_category_mapping,
            p_preferred_lengths, p_length_category_mapping,
            p_preferred_lengths_dresses, p_length_dresses_category_mapping,
            p_preferred_rises,
            -- Type preferences
            p_top_types, p_bottom_types, p_dress_types, p_outerwear_types,
            -- Lifestyle
            p_occasions, p_styles_to_avoid, p_patterns_liked, p_patterns_avoided, p_style_persona,
            -- Brands
            p_preferred_brands, p_brands_to_avoid, p_brand_openness,
            -- Style discovery
            p_style_discovery_complete, p_swiped_items, v_taste_vector,
            -- Metadata
            v_completed_at, NOW()
        )
        ON CONFLICT (anon_id) DO UPDATE SET
            gender = EXCLUDED.gender,
            -- Core setup
            categories = EXCLUDED.categories,
            birthdate = EXCLUDED.birthdate,
            top_sizes = EXCLUDED.top_sizes,
            bottom_sizes = EXCLUDED.bottom_sizes,
            outerwear_sizes = EXCLUDED.outerwear_sizes,
            colors_to_avoid = EXCLUDED.colors_to_avoid,
            materials_to_avoid = EXCLUDED.materials_to_avoid,
            -- Attribute preferences
            preferred_fits = EXCLUDED.preferred_fits,
            fit_category_mapping = EXCLUDED.fit_category_mapping,
            preferred_sleeves = EXCLUDED.preferred_sleeves,
            sleeve_category_mapping = EXCLUDED.sleeve_category_mapping,
            preferred_lengths = EXCLUDED.preferred_lengths,
            length_category_mapping = EXCLUDED.length_category_mapping,
            preferred_lengths_dresses = EXCLUDED.preferred_lengths_dresses,
            length_dresses_category_mapping = EXCLUDED.length_dresses_category_mapping,
            preferred_rises = EXCLUDED.preferred_rises,
            -- Type preferences
            top_types = EXCLUDED.top_types,
            bottom_types = EXCLUDED.bottom_types,
            dress_types = EXCLUDED.dress_types,
            outerwear_types = EXCLUDED.outerwear_types,
            -- Lifestyle
            occasions = EXCLUDED.occasions,
            styles_to_avoid = EXCLUDED.styles_to_avoid,
            patterns_liked = EXCLUDED.patterns_liked,
            patterns_avoided = EXCLUDED.patterns_avoided,
            style_persona = EXCLUDED.style_persona,
            -- Brands
            preferred_brands = EXCLUDED.preferred_brands,
            brands_to_avoid = EXCLUDED.brands_to_avoid,
            brand_openness = EXCLUDED.brand_openness,
            -- Style discovery
            style_discovery_complete = EXCLUDED.style_discovery_complete,
            swiped_items = EXCLUDED.swiped_items,
            taste_vector = EXCLUDED.taste_vector,
            -- Metadata
            completed_at = EXCLUDED.completed_at,
            updated_at = NOW()
        RETURNING id INTO result_id;
    ELSE
        RAISE EXCEPTION 'Either user_id or anon_id must be provided';
    END IF;

    RETURN result_id;
END;
$$;

GRANT EXECUTE ON FUNCTION save_onboarding_profile_v3 TO anon, authenticated;

-- =====================================================
-- STEP 7: Update get_user_recommendation_state to include new fields
-- =====================================================
DROP FUNCTION IF EXISTS get_user_recommendation_state_v3 CASCADE;

CREATE OR REPLACE FUNCTION get_user_recommendation_state_v3(
    p_user_id uuid DEFAULT NULL,
    p_anon_id varchar DEFAULT NULL
)
RETURNS TABLE(
    -- User info
    user_identifier text,
    -- Taste vector
    has_taste_vector boolean,
    taste_vector vector(512),
    -- Onboarding status
    has_onboarding boolean,
    -- Core setup
    onboarding_categories text[],
    birthdate varchar,
    top_sizes text[],
    bottom_sizes text[],
    outerwear_sizes text[],
    colors_to_avoid text[],
    materials_to_avoid text[],
    -- Attribute preferences
    preferred_fits text[],
    fit_category_mapping jsonb,
    preferred_sleeves text[],
    sleeve_category_mapping jsonb,
    preferred_lengths text[],
    length_category_mapping jsonb,
    preferred_lengths_dresses text[],
    length_dresses_category_mapping jsonb,
    preferred_rises text[],
    -- Type preferences
    top_types text[],
    bottom_types text[],
    dress_types text[],
    outerwear_types text[],
    -- Lifestyle
    occasions text[],
    styles_to_avoid text[],
    patterns_liked text[],
    patterns_avoided text[],
    style_persona text[],
    -- Brands
    preferred_brands text[],
    brands_to_avoid text[],
    brand_openness varchar,
    -- Style discovery
    style_discovery_complete boolean,
    swiped_items text[],
    -- State type
    state_type text
)
LANGUAGE plpgsql AS $$
DECLARE
    v_has_onboard boolean := false;
    v_has_taste boolean := false;
    v_taste vector(512);
    v_state text;
BEGIN
    -- Load from onboarding profile
    SELECT
        op.id IS NOT NULL,
        op.taste_vector IS NOT NULL,
        op.taste_vector
    INTO v_has_onboard, v_has_taste, v_taste
    FROM user_onboarding_profiles op
    WHERE (p_user_id IS NOT NULL AND op.user_id = p_user_id)
       OR (p_anon_id IS NOT NULL AND op.anon_id = p_anon_id)
    LIMIT 1;

    -- Fallback to legacy user_seed_preferences if no taste vector in onboarding
    IF NOT v_has_taste THEN
        SELECT sp.taste_vector IS NOT NULL, sp.taste_vector
        INTO v_has_taste, v_taste
        FROM user_seed_preferences sp
        WHERE (p_user_id IS NOT NULL AND sp.user_id = p_user_id)
           OR (p_anon_id IS NOT NULL AND sp.anon_id = p_anon_id)
        LIMIT 1;
    END IF;

    -- Determine state type
    IF v_has_taste THEN
        v_state := 'tinder_complete';
    ELSIF v_has_onboard THEN
        v_state := 'tinder_complete';  -- Has onboarding = use preferences for cold start
    ELSE
        v_state := 'cold_start';
    END IF;

    RETURN QUERY
    SELECT
        COALESCE(p_user_id::text, p_anon_id),
        COALESCE(v_has_taste, false),
        v_taste,
        COALESCE(v_has_onboard, false),
        -- Core setup
        COALESCE(op.categories, '{}'),
        op.birthdate,
        COALESCE(op.top_sizes, '{}'),
        COALESCE(op.bottom_sizes, '{}'),
        COALESCE(op.outerwear_sizes, '{}'),
        COALESCE(op.colors_to_avoid, '{}'),
        COALESCE(op.materials_to_avoid, '{}'),
        -- Attribute preferences
        COALESCE(op.preferred_fits, '{}'),
        COALESCE(op.fit_category_mapping, '[]'::jsonb),
        COALESCE(op.preferred_sleeves, '{}'),
        COALESCE(op.sleeve_category_mapping, '[]'::jsonb),
        COALESCE(op.preferred_lengths, '{}'),
        COALESCE(op.length_category_mapping, '[]'::jsonb),
        COALESCE(op.preferred_lengths_dresses, '{}'),
        COALESCE(op.length_dresses_category_mapping, '[]'::jsonb),
        COALESCE(op.preferred_rises, '{}'),
        -- Type preferences
        COALESCE(op.top_types, '{}'),
        COALESCE(op.bottom_types, '{}'),
        COALESCE(op.dress_types, '{}'),
        COALESCE(op.outerwear_types, '{}'),
        -- Lifestyle
        COALESCE(op.occasions, '{}'),
        COALESCE(op.styles_to_avoid, '{}'),
        COALESCE(op.patterns_liked, '{}'),
        COALESCE(op.patterns_avoided, '{}'),
        COALESCE(op.style_persona, '{}'),
        -- Brands
        COALESCE(op.preferred_brands, '{}'),
        COALESCE(op.brands_to_avoid, '{}'),
        op.brand_openness,
        -- Style discovery
        COALESCE(op.style_discovery_complete, false),
        COALESCE(op.swiped_items, '{}'),
        -- State type
        v_state
    FROM user_onboarding_profiles op
    WHERE (p_user_id IS NOT NULL AND op.user_id = p_user_id)
       OR (p_anon_id IS NOT NULL AND op.anon_id = p_anon_id)
    LIMIT 1;

    -- If no onboarding found, return defaults
    IF NOT FOUND THEN
        RETURN QUERY SELECT
            COALESCE(p_user_id::text, p_anon_id),
            COALESCE(v_has_taste, false),
            v_taste,
            false,  -- has_onboarding
            -- Core setup defaults
            '{}'::text[], NULL::varchar, '{}'::text[], '{}'::text[], '{}'::text[],
            '{}'::text[], '{}'::text[],
            -- Attribute preferences defaults
            '{}'::text[], '[]'::jsonb, '{}'::text[], '[]'::jsonb,
            '{}'::text[], '[]'::jsonb, '{}'::text[], '[]'::jsonb, '{}'::text[],
            -- Type preferences defaults
            '{}'::text[], '{}'::text[], '{}'::text[], '{}'::text[],
            -- Lifestyle defaults
            '{}'::text[], '{}'::text[], '{}'::text[], '{}'::text[], '{}'::text[],
            -- Brands defaults
            '{}'::text[], '{}'::text[], NULL::varchar,
            -- Style discovery defaults
            false, '{}'::text[],
            -- State type
            v_state;
    END IF;
END;
$$;

GRANT EXECUTE ON FUNCTION get_user_recommendation_state_v3 TO anon, authenticated;

-- =====================================================
-- STEP 8: Notify PostgREST to reload schema
-- =====================================================
NOTIFY pgrst, 'reload schema';

SELECT 'Onboarding V3 migration complete - new frontend spec supported' as status;
