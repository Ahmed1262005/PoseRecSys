-- Update get_user_recommendation_state to include lifestyle columns
DROP FUNCTION IF EXISTS get_user_recommendation_state(uuid, varchar);

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
    -- Per-category preferences (JSON)
    tops_prefs jsonb,
    bottoms_prefs jsonb,
    skirts_prefs jsonb,
    dresses_prefs jsonb,
    one_piece_prefs jsonb,
    outerwear_prefs jsonb,
    -- Price preferences
    global_min_price numeric,
    global_max_price numeric,
    -- Lifestyle preferences (NEW)
    styles_to_avoid text[],
    occasions text[],
    patterns_to_avoid text[],
    patterns_preferred text[],
    -- Misc
    sizes text[],
    birthdate text,
    modesty text,
    brand_openness text,
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
    -- Per-category prefs
    v_tops_prefs jsonb;
    v_bottoms_prefs jsonb;
    v_skirts_prefs jsonb;
    v_dresses_prefs jsonb;
    v_one_piece_prefs jsonb;
    v_outerwear_prefs jsonb;
    v_global_min numeric;
    v_global_max numeric;
    -- Lifestyle
    v_styles_to_avoid text[];
    v_occasions text[];
    v_patterns_to_avoid text[];
    v_patterns_preferred text[];
    -- Misc
    v_sizes text[];
    v_birthdate text;
    v_modesty text;
    v_brand_openness text;
BEGIN
    -- Get from onboarding profile
    SELECT
        op.id IS NOT NULL,
        op.taste_vector,
        op.categories,
        op.colors_to_avoid,
        op.materials_to_avoid,
        op.brands_to_avoid,
        op.preferred_brands,
        op.style_directions,
        op.style_discovery,
        op.tops_prefs,
        op.bottoms_prefs,
        op.skirts_prefs,
        op.dresses_prefs,
        op.one_piece_prefs,
        op.outerwear_prefs,
        op.global_min_price,
        op.global_max_price,
        op.styles_to_avoid,
        op.occasions,
        op.patterns_to_avoid,
        op.patterns_preferred,
        op.sizes,
        op.birthdate,
        op.modesty,
        op.brand_openness
    INTO
        v_has_onboard, v_taste, v_onboard_cats, v_colors, v_materials,
        v_brands_avoid, v_brands_prefer, v_styles, v_style_discovery,
        v_tops_prefs, v_bottoms_prefs, v_skirts_prefs, v_dresses_prefs,
        v_one_piece_prefs, v_outerwear_prefs, v_global_min, v_global_max,
        v_styles_to_avoid, v_occasions, v_patterns_to_avoid, v_patterns_preferred,
        v_sizes, v_birthdate, v_modesty, v_brand_openness
    FROM user_onboarding_profiles op
    WHERE (p_user_id IS NOT NULL AND op.user_id = p_user_id)
       OR (p_anon_id IS NOT NULL AND op.anon_id = p_anon_id)
    LIMIT 1;

    -- Check if taste vector exists
    IF v_taste IS NOT NULL THEN
        v_has_taste := true;
        IF v_style_discovery IS NOT NULL AND v_style_discovery ? 'selections' THEN
            SELECT ARRAY_AGG(DISTINCT s->>'category')
            INTO v_tinder_cats
            FROM jsonb_array_elements(v_style_discovery->'selections') s
            WHERE s->>'category' IS NOT NULL;
        END IF;
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
        v_state := 'warm_user';
    ELSIF v_has_onboard THEN
        v_state := 'cold_start';
    ELSE
        v_state := 'anonymous';
    END IF;

    RETURN QUERY SELECT
        COALESCE(p_user_id::text, p_anon_id, 'unknown'),
        v_has_taste,
        v_taste,
        v_tinder_cats,
        v_attr_prefs,
        v_has_onboard,
        v_onboard_cats,
        v_colors,
        v_materials,
        v_brands_avoid,
        v_brands_prefer,
        v_styles,
        v_tops_prefs,
        v_bottoms_prefs,
        v_skirts_prefs,
        v_dresses_prefs,
        v_one_piece_prefs,
        v_outerwear_prefs,
        v_global_min,
        v_global_max,
        v_styles_to_avoid,
        v_occasions,
        v_patterns_to_avoid,
        v_patterns_preferred,
        v_sizes,
        v_birthdate,
        v_modesty,
        v_brand_openness,
        v_state;
END;
$$;

GRANT EXECUTE ON FUNCTION get_user_recommendation_state TO anon, authenticated;

NOTIFY pgrst, 'reload schema';

SELECT 'get_user_recommendation_state updated with lifestyle columns' as status;
