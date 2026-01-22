-- Debug: Test UUID comparison directly
-- Run each section separately to identify the issue

-- Test 1: Does UUID comparison work at all?
SELECT
    'ffffffff-ffff-ffff-ffff-ffffffffffff'::uuid > '00000000-0000-0000-0000-000000000000'::uuid as should_be_true,
    '00000000-0000-0000-0000-000000000001'::uuid < '00000000-0000-0000-0000-000000000002'::uuid as should_also_be_true;

-- Test 2: Check what trending_scores we have
SELECT DISTINCT COALESCE(trending_score, 0.5) as score, COUNT(*)
FROM products
WHERE in_stock = true
GROUP BY COALESCE(trending_score, 0.5)
ORDER BY score DESC
LIMIT 10;

-- Test 3: Direct query with hardcoded cursor (no function)
SELECT COUNT(*) as direct_query_count
FROM products p
WHERE p.in_stock = true
  AND 'female' = ANY(p.gender)
  AND (
      COALESCE(p.trending_score, 0.5) < 0.5
      OR (
          COALESCE(p.trending_score, 0.5) = 0.5
          AND p.id < '00000000-0000-0000-0000-000000000000'::uuid
      )
  );
-- This should return 0!

-- Test 4: Simpler function with just cursor logic
DROP FUNCTION IF EXISTS test_cursor_simple(float, uuid, int);

CREATE FUNCTION test_cursor_simple(
    p_cursor_score float DEFAULT NULL,
    p_cursor_id uuid DEFAULT NULL,
    p_limit int DEFAULT 5
)
RETURNS TABLE(product_id uuid, score float)
LANGUAGE sql AS $$
    SELECT
        p.id as product_id,
        COALESCE(p.trending_score, 0.5)::float as score
    FROM products p
    WHERE p.in_stock = true
      AND 'female' = ANY(p.gender)
      AND (
          p_cursor_score IS NULL
          OR COALESCE(p.trending_score, 0.5) < p_cursor_score
          OR (COALESCE(p.trending_score, 0.5) = p_cursor_score AND p.id < p_cursor_id)
      )
    ORDER BY COALESCE(p.trending_score, 0.5) DESC, p.id DESC
    LIMIT p_limit;
$$;

-- Test 5: Call the simple function
SELECT * FROM test_cursor_simple(0.5, '00000000-0000-0000-0000-000000000000'::uuid, 5);
-- Should return 0 rows!
