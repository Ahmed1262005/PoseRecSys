-- Search Impressions Tracking
--
-- Records which products were shown to each user in search results.
-- Used to soft-demote over-exposed products and improve result freshness.
--
-- Companion RPC: get_user_search_impression_counts()
--   Returns per-product impression counts for a given user,
--   aggregated across all their searches within a time window.

-- =============================================================================
-- Table: search_impressions
-- =============================================================================
CREATE TABLE IF NOT EXISTS search_impressions (
    id          uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id     uuid REFERENCES auth.users(id),
    anon_id     text,
    query       text NOT NULL,
    product_ids uuid[] NOT NULL,
    page        int DEFAULT 1,
    created_at  timestamptz DEFAULT now()
);

-- Index for fast per-user lookups (used by the RPC below)
CREATE INDEX IF NOT EXISTS idx_search_impressions_user_id
    ON search_impressions (user_id, created_at DESC)
    WHERE user_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_search_impressions_anon_id
    ON search_impressions (anon_id, created_at DESC)
    WHERE anon_id IS NOT NULL;

-- Enable RLS (row-level security)
ALTER TABLE search_impressions ENABLE ROW LEVEL SECURITY;

-- Policy: users can only read their own impressions
CREATE POLICY search_impressions_select_own ON search_impressions
    FOR SELECT USING (auth.uid() = user_id);

-- Policy: service role can insert (used by the API server)
CREATE POLICY search_impressions_insert_service ON search_impressions
    FOR INSERT WITH CHECK (true);


-- =============================================================================
-- RPC: get_user_search_impression_counts
--
-- Returns product_id -> impression_count for a user within a time window.
-- Used by the search reranker to soft-demote over-exposed products.
-- =============================================================================
CREATE OR REPLACE FUNCTION get_user_search_impression_counts(
    p_user_id   uuid DEFAULT NULL,
    p_anon_id   text DEFAULT NULL,
    p_since     timestamptz DEFAULT (now() - interval '7 days')
)
RETURNS TABLE (product_id uuid, impression_count bigint)
LANGUAGE sql STABLE
AS $$
    SELECT
        unnested_id AS product_id,
        COUNT(*)    AS impression_count
    FROM
        search_impressions si,
        LATERAL unnest(si.product_ids) AS unnested_id
    WHERE
        si.created_at >= p_since
        AND (
            (p_user_id IS NOT NULL AND si.user_id = p_user_id)
            OR
            (p_anon_id IS NOT NULL AND si.anon_id = p_anon_id)
        )
    GROUP BY unnested_id
    ORDER BY impression_count DESC;
$$;
