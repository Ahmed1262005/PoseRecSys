-- Add lifestyle columns to user_onboarding_profiles
-- Run this to enable lifestyle filtering from saved onboarding profiles

-- Step 1: Add the columns
ALTER TABLE user_onboarding_profiles
ADD COLUMN IF NOT EXISTS styles_to_avoid text[] DEFAULT '{}';

ALTER TABLE user_onboarding_profiles
ADD COLUMN IF NOT EXISTS occasions text[] DEFAULT '{}';

ALTER TABLE user_onboarding_profiles
ADD COLUMN IF NOT EXISTS patterns_to_avoid text[] DEFAULT '{}';

ALTER TABLE user_onboarding_profiles
ADD COLUMN IF NOT EXISTS patterns_preferred text[] DEFAULT '{}';

COMMENT ON COLUMN user_onboarding_profiles.styles_to_avoid IS
    'Coverage styles to exclude: deep-necklines, sheer, cutouts, backless, strapless';
COMMENT ON COLUMN user_onboarding_profiles.occasions IS
    'Occasions to include: casual, office, evening, beach';
COMMENT ON COLUMN user_onboarding_profiles.patterns_to_avoid IS
    'Patterns to exclude: floral, animal-print, stripes, etc.';
COMMENT ON COLUMN user_onboarding_profiles.patterns_preferred IS
    'Preferred patterns: solid, minimal, geometric, etc.';

SELECT 'Lifestyle columns added to user_onboarding_profiles' as status;

-- NOTE: The save RPC function update is commented out below.
-- Run this separately if you want to update the existing save function.
-- You'll need to drop the old function first with its exact signature.

/*
-- To update save function, first find and drop existing:
-- SELECT proname, pg_get_function_arguments(oid) FROM pg_proc WHERE proname = 'save_onboarding_profile_v2';
-- DROP FUNCTION save_onboarding_profile_v2(<exact argument list>);

CREATE OR REPLACE FUNCTION save_onboarding_profile_v2(
    p_user_id uuid DEFAULT NULL,
    p_anon_id text DEFAULT NULL,
    p_gender text DEFAULT 'female',
    p_categories text[] DEFAULT '{}',
    p_sizes text[] DEFAULT '{}',
    p_birthdate text DEFAULT NULL,
    p_colors_to_avoid text[] DEFAULT '{}',
    p_materials_to_avoid text[] DEFAULT '{}',
    p_tops_prefs jsonb DEFAULT '{}',
    p_bottoms_prefs jsonb DEFAULT '{}',
    p_skirts_prefs jsonb DEFAULT '{}',
    p_dresses_prefs jsonb DEFAULT '{}',
    p_one_piece_prefs jsonb DEFAULT '{}',
    p_outerwear_prefs jsonb DEFAULT '{}',
    p_style_directions text[] DEFAULT '{}',
    p_modesty text DEFAULT NULL,
    p_preferred_brands text[] DEFAULT '{}',
    p_brands_to_avoid text[] DEFAULT '{}',
    p_brand_openness text DEFAULT NULL,
    p_global_min_price numeric DEFAULT NULL,
    p_global_max_price numeric DEFAULT NULL,
    p_style_discovery jsonb DEFAULT NULL,
    p_taste_vector float[] DEFAULT NULL,
    p_completed_at text DEFAULT NULL,
    -- NEW: Lifestyle fields
    p_styles_to_avoid text[] DEFAULT '{}',
    p_occasions text[] DEFAULT '{}',
    p_patterns_to_avoid text[] DEFAULT '{}',
    p_patterns_preferred text[] DEFAULT '{}'
)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_profile_id uuid;
    v_result jsonb;
BEGIN
    -- Upsert based on user_id or anon_id
    IF p_user_id IS NOT NULL THEN
        INSERT INTO user_onboarding_profiles (
            user_id, gender, categories, sizes, birthdate,
            colors_to_avoid, materials_to_avoid,
            tops_prefs, bottoms_prefs, skirts_prefs, dresses_prefs,
            one_piece_prefs, outerwear_prefs,
            style_directions, modesty, preferred_brands, brands_to_avoid,
            brand_openness, global_min_price, global_max_price,
            style_discovery, taste_vector, completed_at,
            styles_to_avoid, occasions, patterns_to_avoid, patterns_preferred,
            updated_at
        ) VALUES (
            p_user_id, p_gender, p_categories, p_sizes, p_birthdate,
            p_colors_to_avoid, p_materials_to_avoid,
            p_tops_prefs, p_bottoms_prefs, p_skirts_prefs, p_dresses_prefs,
            p_one_piece_prefs, p_outerwear_prefs,
            p_style_directions, p_modesty, p_preferred_brands, p_brands_to_avoid,
            p_brand_openness, p_global_min_price, p_global_max_price,
            p_style_discovery, p_taste_vector, p_completed_at,
            p_styles_to_avoid, p_occasions, p_patterns_to_avoid, p_patterns_preferred,
            now()
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
            updated_at = now()
        RETURNING id INTO v_profile_id;
    ELSIF p_anon_id IS NOT NULL THEN
        INSERT INTO user_onboarding_profiles (
            anon_id, gender, categories, sizes, birthdate,
            colors_to_avoid, materials_to_avoid,
            tops_prefs, bottoms_prefs, skirts_prefs, dresses_prefs,
            one_piece_prefs, outerwear_prefs,
            style_directions, modesty, preferred_brands, brands_to_avoid,
            brand_openness, global_min_price, global_max_price,
            style_discovery, taste_vector, completed_at,
            styles_to_avoid, occasions, patterns_to_avoid, patterns_preferred,
            updated_at
        ) VALUES (
            p_anon_id, p_gender, p_categories, p_sizes, p_birthdate,
            p_colors_to_avoid, p_materials_to_avoid,
            p_tops_prefs, p_bottoms_prefs, p_skirts_prefs, p_dresses_prefs,
            p_one_piece_prefs, p_outerwear_prefs,
            p_style_directions, p_modesty, p_preferred_brands, p_brands_to_avoid,
            p_brand_openness, p_global_min_price, p_global_max_price,
            p_style_discovery, p_taste_vector, p_completed_at,
            p_styles_to_avoid, p_occasions, p_patterns_to_avoid, p_patterns_preferred,
            now()
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
            updated_at = now()
        RETURNING id INTO v_profile_id;
    ELSE
        RETURN jsonb_build_object('status', 'error', 'message', 'Either user_id or anon_id required');
    END IF;

    RETURN jsonb_build_object(
        'status', 'success',
        'profile_id', v_profile_id,
        'user_id', p_user_id,
        'anon_id', p_anon_id
    );
END;
$$;

GRANT EXECUTE ON FUNCTION save_onboarding_profile_v2 TO anon, authenticated;
*/
