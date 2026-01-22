-- User interactions table for tracking explicit engagement actions
-- This enables:
-- 1. Updating taste_vector based on positive signals (wishlist, cart)
-- 2. Training SASRec on user sequences
-- 3. Analytics on user behavior
--
-- Actions tracked:
-- - click: User taps to view product details
-- - hover: User swipes through photo gallery
-- - add_to_wishlist: Strong positive signal (like/save)
-- - add_to_cart: Conversion intent
-- - purchase: Conversion
--
-- NOTE: view/skip are NOT tracked here - they're implicit from session seen_ids

CREATE TABLE IF NOT EXISTS user_interactions (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,

    -- User identification (one of these required)
    user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
    anon_id text,

    -- Session context
    session_id text,

    -- The interaction
    product_id uuid NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    action text NOT NULL CHECK (action IN ('click', 'hover', 'add_to_wishlist', 'add_to_cart', 'purchase')),

    -- Context
    source text,  -- 'feed', 'search', 'similar', 'style-this'
    position int,  -- Position in feed when interacted

    -- Timestamps
    created_at timestamptz DEFAULT now(),

    -- Ensure we have at least one user identifier
    CONSTRAINT user_identifier_required CHECK (user_id IS NOT NULL OR anon_id IS NOT NULL)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_interactions_anon_id ON user_interactions(anon_id) WHERE anon_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON user_interactions(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_interactions_product_id ON user_interactions(product_id);
CREATE INDEX IF NOT EXISTS idx_interactions_created_at ON user_interactions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_interactions_action ON user_interactions(action);

-- Composite index for user history queries
CREATE INDEX IF NOT EXISTS idx_interactions_user_history
    ON user_interactions(anon_id, created_at DESC)
    WHERE anon_id IS NOT NULL;

-- Enable RLS
ALTER TABLE user_interactions ENABLE ROW LEVEL SECURITY;

-- Policy: Users can insert their own interactions
CREATE POLICY "Users can insert own interactions" ON user_interactions
    FOR INSERT WITH CHECK (true);

-- Policy: Users can read their own interactions
CREATE POLICY "Users can read own interactions" ON user_interactions
    FOR SELECT USING (
        anon_id = current_setting('request.headers', true)::json->>'x-anon-id'
        OR user_id = auth.uid()
    );

-- Grant access
GRANT SELECT, INSERT ON user_interactions TO anon, authenticated;

-- Helper function to get user's positive interactions (for taste vector update)
CREATE OR REPLACE FUNCTION get_user_positive_products(
    p_anon_id text DEFAULT NULL,
    p_user_id uuid DEFAULT NULL,
    p_limit int DEFAULT 50
)
RETURNS TABLE(product_id uuid, action text, created_at timestamptz)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT ui.product_id, ui.action, ui.created_at
    FROM user_interactions ui
    WHERE ui.action IN ('add_to_wishlist', 'add_to_cart', 'purchase')
      AND (
          (p_anon_id IS NOT NULL AND ui.anon_id = p_anon_id)
          OR (p_user_id IS NOT NULL AND ui.user_id = p_user_id)
      )
    ORDER BY ui.created_at DESC
    LIMIT p_limit;
END;
$$;

GRANT EXECUTE ON FUNCTION get_user_positive_products TO anon, authenticated;

-- Verify
SELECT 'user_interactions table created' as status;
