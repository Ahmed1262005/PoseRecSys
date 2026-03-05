-- POSE Canvas: user inspiration images + embeddings
-- Stores uploaded/URL/Pinterest inspiration images with FashionCLIP embeddings
-- for style extraction and taste vector computation.

-- =====================================================
-- 1. Inspirations table
-- =====================================================
CREATE TABLE IF NOT EXISTS user_inspirations (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

    -- Source metadata
    source text NOT NULL CHECK (source IN ('upload', 'url', 'camera', 'pinterest')),
    image_url text NOT NULL,
    original_url text,            -- original URL before any re-hosting
    title text,

    -- FashionCLIP embedding (512-dim, L2-normalized)
    embedding vector(512) NOT NULL,

    -- Style classification (from nearest-neighbor aggregation)
    style_label text,             -- dominant style tag (e.g. 'Boho', 'Classic')
    style_confidence float,       -- confidence of dominant label (0-1)
    style_attributes jsonb DEFAULT '{}'::jsonb,
    -- Example style_attributes:
    -- {
    --   "style_tags": {"Boho": 0.45, "Romantic": 0.30, "Classic": 0.15},
    --   "pattern": {"floral": 0.50, "solid": 0.35},
    --   "color_family": {"Neutrals": 0.40, "Browns": 0.35},
    --   "formality": {"Casual": 0.60, "Smart Casual": 0.25},
    --   "occasions": {"casual": 0.55, "evening": 0.30},
    --   "silhouette": {"A-Line": 0.40, "Fitted": 0.35},
    --   "fit_type": {"regular": 0.50, "slim": 0.30},
    --   "sleeve_type": {"long": 0.45, "short": 0.35},
    --   "neckline": {"v-neck": 0.40, "crew": 0.30}
    -- }

    -- Pinterest-specific fields
    pinterest_pin_id text,

    created_at timestamptz DEFAULT NOW(),
    updated_at timestamptz DEFAULT NOW()
);

-- =====================================================
-- 2. Indexes
-- =====================================================

-- Fast lookup by user
CREATE INDEX IF NOT EXISTS idx_user_inspirations_user_id
    ON user_inspirations(user_id);

-- Prevent duplicate Pinterest pins per user
CREATE UNIQUE INDEX IF NOT EXISTS idx_user_inspirations_user_pin
    ON user_inspirations(user_id, pinterest_pin_id)
    WHERE pinterest_pin_id IS NOT NULL;

-- HNSW index for embedding similarity (matches image_embeddings config)
CREATE INDEX IF NOT EXISTS idx_user_inspirations_embedding_hnsw
    ON user_inspirations
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- =====================================================
-- 3. Row-Level Security
-- =====================================================
ALTER TABLE user_inspirations ENABLE ROW LEVEL SECURITY;

-- Users can only read their own inspirations
CREATE POLICY user_inspirations_select ON user_inspirations
    FOR SELECT USING (auth.uid() = user_id);

-- Users can only insert their own inspirations
CREATE POLICY user_inspirations_insert ON user_inspirations
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Users can only delete their own inspirations
CREATE POLICY user_inspirations_delete ON user_inspirations
    FOR DELETE USING (auth.uid() = user_id);

-- Users can only update their own inspirations
CREATE POLICY user_inspirations_update ON user_inspirations
    FOR UPDATE USING (auth.uid() = user_id);

-- =====================================================
-- 4. Updated-at trigger
-- =====================================================
CREATE OR REPLACE FUNCTION update_user_inspirations_updated_at()
RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_user_inspirations_updated_at
    BEFORE UPDATE ON user_inspirations
    FOR EACH ROW
    EXECUTE FUNCTION update_user_inspirations_updated_at();
