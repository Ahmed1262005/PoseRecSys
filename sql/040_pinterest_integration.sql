-- Pinterest OAuth token storage + taste vector update helper

-- =====================================================
-- 1. OAuth token storage
-- =====================================================
CREATE TABLE IF NOT EXISTS user_oauth_tokens (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    provider text NOT NULL,
    access_token text NOT NULL,
    refresh_token text,
    token_type text,
    scope text,
    expires_at timestamptz,
    refresh_expires_at timestamptz,
    account_id text,
    account_username text,
    metadata jsonb DEFAULT '{}'::jsonb,
    last_sync_at timestamptz,
    last_sync_count integer,
    last_sync_error text,
    created_at timestamptz DEFAULT NOW(),
    updated_at timestamptz DEFAULT NOW(),
    UNIQUE (user_id, provider)
);

CREATE INDEX IF NOT EXISTS idx_user_oauth_tokens_user_provider
    ON user_oauth_tokens(user_id, provider);

ALTER TABLE user_oauth_tokens ENABLE ROW LEVEL SECURITY;

-- =====================================================
-- 2. Update taste vector without overwriting onboarding
-- =====================================================
CREATE OR REPLACE FUNCTION update_user_taste_vector(
    p_user_id uuid,
    p_taste_vector float8[]
)
RETURNS void
LANGUAGE plpgsql AS $$
DECLARE
    v_taste_vector vector(512);
BEGIN
    IF p_taste_vector IS NULL OR COALESCE(array_length(p_taste_vector, 1), 0) <> 512 THEN
        RAISE EXCEPTION 'taste_vector must be length 512';
    END IF;

    v_taste_vector := p_taste_vector::vector(512);

    INSERT INTO user_onboarding_profiles (
        user_id,
        taste_vector,
        style_discovery_complete,
        completed_at,
        updated_at
    )
    VALUES (
        p_user_id,
        v_taste_vector,
        true,
        NOW(),
        NOW()
    )
    ON CONFLICT (user_id) DO UPDATE SET
        taste_vector = EXCLUDED.taste_vector,
        style_discovery_complete = true,
        updated_at = NOW(),
        completed_at = COALESCE(user_onboarding_profiles.completed_at, EXCLUDED.completed_at);
END;
$$;
