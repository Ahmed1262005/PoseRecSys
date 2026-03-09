-- ============================================================
-- 074: Increase statement_timeout for search functions
--
-- Under concurrent load (4-6 parallel pgvector queries),
-- Supabase's default statement_timeout (~8s) kills vector
-- scans before they complete.  Setting 30s per-function gives
-- ample headroom without affecting other queries.
-- ============================================================

-- search_multimodal — primary multimodal vector search
ALTER FUNCTION search_multimodal
    SET statement_timeout = '30s';

-- search_semantic_with_attributes — attribute-filtered vector search
ALTER FUNCTION search_semantic_with_attributes
    SET statement_timeout = '30s';

-- text_search_products — legacy image-only vector search (fallback)
ALTER FUNCTION text_search_products
    SET statement_timeout = '30s';
