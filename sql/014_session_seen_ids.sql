-- Session seen IDs table for ML training data
-- This stores which products were shown to users in each session
-- Used for negative sampling: items shown but not interacted with = implicit negatives
--
-- This is a batch-sync table - frontend sends seen_ids periodically (every N pages)
-- rather than on every view, for performance.

CREATE TABLE IF NOT EXISTS session_seen_ids (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,

    -- User identification (one of these required)
    user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
    anon_id text,

    -- Session context
    session_id text NOT NULL,

    -- Seen products (batch of product IDs shown in this session)
    seen_ids uuid[] NOT NULL,  -- Array of product IDs

    -- Timestamps
    synced_at timestamptz DEFAULT now(),

    -- Ensure we have at least one user identifier
    CONSTRAINT session_user_identifier_required CHECK (user_id IS NOT NULL OR anon_id IS NOT NULL)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_session_seen_ids_session ON session_seen_ids(session_id);
CREATE INDEX IF NOT EXISTS idx_session_seen_ids_anon ON session_seen_ids(anon_id) WHERE anon_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_session_seen_ids_user ON session_seen_ids(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_session_seen_ids_synced_at ON session_seen_ids(synced_at DESC);

-- Enable RLS
ALTER TABLE session_seen_ids ENABLE ROW LEVEL SECURITY;

-- Policy: Users can insert their own session data
CREATE POLICY "Users can insert own session_seen_ids" ON session_seen_ids
    FOR INSERT WITH CHECK (true);

-- Policy: Users can read their own session data
CREATE POLICY "Users can read own session_seen_ids" ON session_seen_ids
    FOR SELECT USING (
        anon_id = current_setting('request.headers', true)::json->>'x-anon-id'
        OR user_id = auth.uid()
    );

-- Grant access
GRANT SELECT, INSERT ON session_seen_ids TO anon, authenticated;

-- Helper function to get all seen products for a user (for training)
CREATE OR REPLACE FUNCTION get_user_seen_products(
    p_anon_id text DEFAULT NULL,
    p_user_id uuid DEFAULT NULL,
    p_since timestamptz DEFAULT NULL
)
RETURNS TABLE(session_id text, product_id uuid, synced_at timestamptz)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT s.session_id, unnest(s.seen_ids) as product_id, s.synced_at
    FROM session_seen_ids s
    WHERE (
        (p_anon_id IS NOT NULL AND s.anon_id = p_anon_id)
        OR (p_user_id IS NOT NULL AND s.user_id = p_user_id)
    )
    AND (p_since IS NULL OR s.synced_at >= p_since)
    ORDER BY s.synced_at DESC;
END;
$$;

GRANT EXECUTE ON FUNCTION get_user_seen_products TO anon, authenticated;

-- Helper function to get training data (positives + negatives) for a user
CREATE OR REPLACE FUNCTION get_user_training_data(
    p_anon_id text DEFAULT NULL,
    p_user_id uuid DEFAULT NULL,
    p_since timestamptz DEFAULT NULL
)
RETURNS TABLE(
    session_id text,
    product_id uuid,
    label text,  -- 'positive' or 'negative'
    action text,  -- action type or 'implicit_view' for negatives
    event_time timestamptz
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    -- Positives: explicit interactions
    SELECT
        ui.session_id,
        ui.product_id,
        'positive'::text as label,
        ui.action,
        ui.created_at as event_time
    FROM user_interactions ui
    WHERE ui.action IN ('add_to_wishlist', 'add_to_cart', 'purchase')
      AND (
          (p_anon_id IS NOT NULL AND ui.anon_id = p_anon_id)
          OR (p_user_id IS NOT NULL AND ui.user_id = p_user_id)
      )
      AND (p_since IS NULL OR ui.created_at >= p_since)

    UNION ALL

    -- Negatives: seen but not positively interacted
    SELECT
        s.session_id,
        unnest(s.seen_ids) as product_id,
        'negative'::text as label,
        'implicit_view'::text as action,
        s.synced_at as event_time
    FROM session_seen_ids s
    WHERE (
        (p_anon_id IS NOT NULL AND s.anon_id = p_anon_id)
        OR (p_user_id IS NOT NULL AND s.user_id = p_user_id)
    )
    AND (p_since IS NULL OR s.synced_at >= p_since)
    AND NOT EXISTS (
        SELECT 1 FROM user_interactions ui
        WHERE ui.session_id = s.session_id
        AND ui.product_id = ANY(s.seen_ids)
        AND ui.action IN ('add_to_wishlist', 'add_to_cart', 'purchase')
    )
    ORDER BY event_time DESC;
END;
$$;

GRANT EXECUTE ON FUNCTION get_user_training_data TO anon, authenticated;

-- Verify
SELECT 'session_seen_ids table created' as status;
