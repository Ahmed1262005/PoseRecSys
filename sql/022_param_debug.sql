-- Debug: Check if parameters are received correctly
CREATE OR REPLACE FUNCTION test_param_debug(
    exclude_styles text[] DEFAULT NULL,
    style_threshold float DEFAULT 0.25
)
RETURNS TABLE(
    param_exclude_styles text,
    param_threshold float,
    array_length int,
    first_element text
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        exclude_styles::text as param_exclude_styles,
        style_threshold as param_threshold,
        array_length(exclude_styles, 1) as array_length,
        exclude_styles[1] as first_element;
END;
$$;

GRANT EXECUTE ON FUNCTION test_param_debug TO anon, authenticated;

SELECT 'Param debug function created' as status;
