-- =====================================================
-- Drop ALL existing recommendation functions
-- This uses dynamic SQL to find and drop all overloads
-- Run this FIRST, then run 003b_create_functions.sql
-- =====================================================

DO $$
DECLARE
    func_record RECORD;
    drop_cmd TEXT;
BEGIN
    -- Find and drop all functions with these names
    FOR func_record IN
        SELECT
            p.proname,
            pg_get_function_identity_arguments(p.oid) as args
        FROM pg_proc p
        JOIN pg_namespace n ON p.pronamespace = n.oid
        WHERE n.nspname = 'public'
        AND p.proname IN (
            'get_trending_products',
            'match_embeddings',
            'match_products_by_embedding',
            'get_similar_products',
            'get_product_categories',
            'get_product_embedding',
            'get_popular_products',
            'search_products_by_category',
            'save_tinder_preferences'
        )
    LOOP
        drop_cmd := 'DROP FUNCTION IF EXISTS public.' || func_record.proname || '(' || func_record.args || ') CASCADE';
        RAISE NOTICE 'Dropping: %', drop_cmd;
        EXECUTE drop_cmd;
    END LOOP;
END;
$$;

-- Verify they're all gone
SELECT proname, pg_get_function_arguments(oid) as args
FROM pg_proc
WHERE proname IN (
    'get_trending_products',
    'match_embeddings',
    'match_products_by_embedding',
    'get_similar_products',
    'get_product_categories'
)
AND pronamespace = 'public'::regnamespace;
