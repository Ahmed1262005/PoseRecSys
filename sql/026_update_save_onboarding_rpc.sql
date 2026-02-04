-- Update save_onboarding_profile_v2 to include lifestyle parameters
-- This fixes the error when trying to save with styles_to_avoid, occasions, etc.

-- Step 1: Drop ALL versions of the function (CASCADE handles overloads)
DROP FUNCTION IF EXISTS save_onboarding_profile_v2 CASCADE;

-- Step 2: Recreate with lifestyle parameters
CREATE OR REPLACE FUNCTION save_onboarding_profile_v2(
    p_user_id uuid DEFAULT NULL,
    p_anon_id varchar DEFAULT NULL,
    p_gender varchar DEFAULT 'female',
    p_categories text[] DEFAULT '{}',
    p_sizes text[] DEFAULT '{}',
    p_birthdate varchar DEFAULT NULL,
    p_colors_to_avoid text[] DEFAULT '{}',
    p_materials_to_avoid text[] DEFAULT '{}',
    p_tops_prefs jsonb DEFAULT '{}',
    p_bottoms_prefs jsonb DEFAULT '{}',
    p_skirts_prefs jsonb DEFAULT '{}',
    p_dresses_prefs jsonb DEFAULT '{}',
    p_one_piece_prefs jsonb DEFAULT '{}',
    p_outerwear_prefs jsonb DEFAULT '{}',
    p_style_directions text[] DEFAULT '{}',
    p_modesty varchar DEFAULT NULL,
    p_preferred_brands text[] DEFAULT '{}',
    p_brands_to_avoid text[] DEFAULT '{}',
    p_brand_openness varchar DEFAULT NULL,
    p_global_min_price numeric DEFAULT NULL,
    p_global_max_price numeric DEFAULT NULL,
    p_style_discovery jsonb DEFAULT NULL,
    p_taste_vector float8[] DEFAULT NULL,
    p_completed_at varchar DEFAULT NULL,
    p_styles_to_avoid text[] DEFAULT '{}',
    p_occasions text[] DEFAULT '{}',
    p_patterns_to_avoid text[] DEFAULT '{}',
    p_patterns_preferred text[] DEFAULT '{}'
)
RETURNS uuid
LANGUAGE plpgsql AS $$
DECLARE
    result_id uuid;
    v_taste_vector vector(512);
    v_completed_at timestamptz;
BEGIN
    IF p_taste_vector IS NOT NULL AND array_length(p_taste_vector, 1) = 512 THEN
        v_taste_vector := p_taste_vector::vector(512);
    ELSE
        v_taste_vector := NULL;
    END IF;

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
            user_id, gender, categories, sizes, birthdate, colors_to_avoid, materials_to_avoid,
            tops_prefs, bottoms_prefs, skirts_prefs, dresses_prefs, one_piece_prefs, outerwear_prefs,
            style_directions, modesty, preferred_brands, brands_to_avoid, brand_openness,
            global_min_price, global_max_price, style_discovery, taste_vector, completed_at,
            styles_to_avoid, occasions, patterns_to_avoid, patterns_preferred, updated_at
        )
        VALUES (
            p_user_id, p_gender, p_categories, p_sizes, p_birthdate, p_colors_to_avoid, p_materials_to_avoid,
            p_tops_prefs, p_bottoms_prefs, p_skirts_prefs, p_dresses_prefs, p_one_piece_prefs, p_outerwear_prefs,
            p_style_directions, p_modesty, p_preferred_brands, p_brands_to_avoid, p_brand_openness,
            p_global_min_price, p_global_max_price, COALESCE(p_style_discovery, '{}'), v_taste_vector, v_completed_at,
            COALESCE(p_styles_to_avoid, '{}'), COALESCE(p_occasions, '{}'),
            COALESCE(p_patterns_to_avoid, '{}'), COALESCE(p_patterns_preferred, '{}'), NOW()
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
            styles_to_avoid = EXCLUDED.styles_to_avoid,
            occasions = EXCLUDED.occasions,
            patterns_to_avoid = EXCLUDED.patterns_to_avoid,
            patterns_preferred = EXCLUDED.patterns_preferred,
            updated_at = NOW()
        RETURNING id INTO result_id;
    ELSIF p_anon_id IS NOT NULL THEN
        INSERT INTO user_onboarding_profiles (
            anon_id, gender, categories, sizes, birthdate, colors_to_avoid, materials_to_avoid,
            tops_prefs, bottoms_prefs, skirts_prefs, dresses_prefs, one_piece_prefs, outerwear_prefs,
            style_directions, modesty, preferred_brands, brands_to_avoid, brand_openness,
            global_min_price, global_max_price, style_discovery, taste_vector, completed_at,
            styles_to_avoid, occasions, patterns_to_avoid, patterns_preferred, updated_at
        )
        VALUES (
            p_anon_id, p_gender, p_categories, p_sizes, p_birthdate, p_colors_to_avoid, p_materials_to_avoid,
            p_tops_prefs, p_bottoms_prefs, p_skirts_prefs, p_dresses_prefs, p_one_piece_prefs, p_outerwear_prefs,
            p_style_directions, p_modesty, p_preferred_brands, p_brands_to_avoid, p_brand_openness,
            p_global_min_price, p_global_max_price, COALESCE(p_style_discovery, '{}'), v_taste_vector, v_completed_at,
            COALESCE(p_styles_to_avoid, '{}'), COALESCE(p_occasions, '{}'),
            COALESCE(p_patterns_to_avoid, '{}'), COALESCE(p_patterns_preferred, '{}'), NOW()
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
            styles_to_avoid = EXCLUDED.styles_to_avoid,
            occasions = EXCLUDED.occasions,
            patterns_to_avoid = EXCLUDED.patterns_to_avoid,
            patterns_preferred = EXCLUDED.patterns_preferred,
            updated_at = NOW()
        RETURNING id INTO result_id;
    ELSE
        RAISE EXCEPTION 'Either user_id or anon_id must be provided';
    END IF;

    RETURN result_id;
END;
$$;

GRANT EXECUTE ON FUNCTION save_onboarding_profile_v2 TO anon, authenticated;

NOTIFY pgrst, 'reload schema';

SELECT 'save_onboarding_profile_v2 updated with lifestyle parameters' as status;
