-- =====================================================
-- Recommendation System Tables Migration
-- Run this in Supabase SQL Editor
-- =====================================================

-- =====================================================
-- STEP 1: Create user_seed_preferences table
-- Stores Tinder-style test results for cold start
-- =====================================================
CREATE TABLE IF NOT EXISTS user_seed_preferences (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- User identification (one required)
    user_id         UUID UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
    anon_id         VARCHAR(64) UNIQUE,

    -- Tinder test metadata
    tinder_session_id   VARCHAR(64),
    gender              VARCHAR(16) NOT NULL DEFAULT 'female',
    rounds_completed    INTEGER DEFAULT 0,
    categories_tested   TEXT[] DEFAULT '{}',

    -- Preference summary (from Tinder test)
    attribute_preferences JSONB NOT NULL DEFAULT '{}',
    /*
    Structure:
    {
        "pattern": {
            "preferred": [["solid", 0.78], ["floral", 0.65]],
            "avoided": [["animal_print", 0.22]]
        },
        "style": {...},
        "color_family": {...},
        ...
    }
    */

    -- Cluster preferences (optional)
    cluster_preferences JSONB DEFAULT '{}',

    -- Quality metrics
    prediction_accuracy FLOAT,

    -- Raw 512-dim taste vector (for pgvector search)
    taste_vector        vector(512),

    -- Timestamps
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT seed_prefs_has_identity CHECK (user_id IS NOT NULL OR anon_id IS NOT NULL)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_seed_prefs_user ON user_seed_preferences (user_id) WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_seed_prefs_anon ON user_seed_preferences (anon_id) WHERE anon_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_seed_prefs_gender ON user_seed_preferences (gender);
CREATE INDEX IF NOT EXISTS idx_seed_prefs_completed ON user_seed_preferences (completed_at DESC) WHERE completed_at IS NOT NULL;

-- Vector index for taste_vector similarity search
CREATE INDEX IF NOT EXISTS idx_seed_prefs_taste_vector ON user_seed_preferences
    USING ivfflat (taste_vector vector_cosine_ops) WITH (lists = 50);

-- =====================================================
-- STEP 2: Create user_vectors table
-- Cached computed vectors for recommendations
-- =====================================================
CREATE TABLE IF NOT EXISTS user_vectors (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- User identification (one required)
    user_id         UUID UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
    anon_id         VARCHAR(64) UNIQUE,

    -- The seed vector for similarity search
    seed_vector     vector(512),

    -- Metadata
    vector_source   VARCHAR(32) NOT NULL,  -- 'tinder', 'behavior', 'hybrid'
    vector_version  VARCHAR(32) NOT NULL DEFAULT 'v1.0',

    -- Component weights (for hybrid vectors)
    component_weights JSONB DEFAULT '{}',

    -- Timestamps
    computed_at     TIMESTAMPTZ DEFAULT NOW(),
    expires_at      TIMESTAMPTZ,

    CONSTRAINT vectors_has_identity CHECK (user_id IS NOT NULL OR anon_id IS NOT NULL)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_user_vectors_user ON user_vectors (user_id) WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_user_vectors_anon ON user_vectors (anon_id) WHERE anon_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_user_vectors_source ON user_vectors (vector_source);

-- Vector index for seed_vector similarity search
CREATE INDEX IF NOT EXISTS idx_user_vectors_embedding ON user_vectors
    USING ivfflat (seed_vector vector_cosine_ops) WITH (lists = 50);

-- =====================================================
-- STEP 3: Create sku_popularity table
-- Aggregated popularity metrics for trending
-- =====================================================
CREATE TABLE IF NOT EXISTS sku_popularity (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sku_id          UUID UNIQUE NOT NULL,  -- References products.id

    -- Raw counts
    view_count      INTEGER DEFAULT 0,
    click_count     INTEGER DEFAULT 0,
    cart_count      INTEGER DEFAULT 0,
    purchase_count  INTEGER DEFAULT 0,
    like_count      INTEGER DEFAULT 0,
    dislike_count   INTEGER DEFAULT 0,

    -- Computed scores
    popularity_score    FLOAT DEFAULT 0.0,
    trending_score      FLOAT DEFAULT 0.0,
    conversion_rate     FLOAT DEFAULT 0.0,

    -- Time-windowed counts (last 7 days)
    views_7d        INTEGER DEFAULT 0,
    clicks_7d       INTEGER DEFAULT 0,
    purchases_7d    INTEGER DEFAULT 0,

    -- Timestamps
    last_viewed_at  TIMESTAMPTZ,
    last_purchased_at TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_popularity_sku ON sku_popularity (sku_id);
CREATE INDEX IF NOT EXISTS idx_popularity_score ON sku_popularity (popularity_score DESC);
CREATE INDEX IF NOT EXISTS idx_popularity_trending ON sku_popularity (trending_score DESC);

-- =====================================================
-- STEP 4: Create match_embeddings function
-- For pgvector similarity search
-- =====================================================
CREATE OR REPLACE FUNCTION match_embeddings(
    query_embedding vector(512),
    match_count int DEFAULT 10,
    filter_gender text DEFAULT NULL
)
RETURNS TABLE (
    id uuid,
    review_sku_id uuid,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        ie.id,
        ie.review_sku_id,
        1 - (ie.embedding <=> query_embedding) as similarity
    FROM image_embeddings ie
    ORDER BY ie.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- =====================================================
-- STEP 5: Create match_products_by_embedding function
-- Returns full product info with similarity scores
-- =====================================================
CREATE OR REPLACE FUNCTION match_products_by_embedding(
    query_embedding vector(512),
    match_count int DEFAULT 20,
    filter_gender text DEFAULT NULL,
    filter_category text DEFAULT NULL
)
RETURNS TABLE (
    product_id uuid,
    name text,
    brand text,
    category text,
    gender text[],
    price numeric,
    primary_image_url text,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id as product_id,
        p.name,
        p.brand,
        p.category,
        p.gender,
        p.price,
        p.primary_image_url,
        1 - (ie.embedding <=> query_embedding) as similarity
    FROM image_embeddings ie
    JOIN products p ON ie.review_sku_id = p.id
    WHERE
        (filter_gender IS NULL OR filter_gender = ANY(p.gender))
        AND (filter_category IS NULL OR p.category = filter_category)
        AND p.in_stock = true
    ORDER BY ie.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- =====================================================
-- STEP 6: Create get_trending_products function
-- For fallback when no personalization available
-- =====================================================
CREATE OR REPLACE FUNCTION get_trending_products(
    filter_gender text DEFAULT NULL,
    filter_category text DEFAULT NULL,
    result_limit int DEFAULT 50
)
RETURNS TABLE (
    product_id uuid,
    name text,
    brand text,
    category text,
    gender text[],
    price numeric,
    primary_image_url text,
    trending_score float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id as product_id,
        p.name,
        p.brand,
        p.category,
        p.gender,
        p.price,
        p.primary_image_url,
        p.trending_score::float
    FROM products p
    WHERE
        (filter_gender IS NULL OR filter_gender = ANY(p.gender))
        AND (filter_category IS NULL OR p.category = filter_category)
        AND p.in_stock = true
    ORDER BY p.trending_score DESC, p.created_at DESC
    LIMIT result_limit;
END;
$$;

-- =====================================================
-- STEP 7: Create save_tinder_preferences function
-- Upserts user preferences from Tinder test
-- =====================================================
CREATE OR REPLACE FUNCTION save_tinder_preferences(
    p_user_id uuid DEFAULT NULL,
    p_anon_id varchar DEFAULT NULL,
    p_gender varchar DEFAULT 'female',
    p_rounds_completed int DEFAULT 0,
    p_categories_tested text[] DEFAULT '{}',
    p_attribute_preferences jsonb DEFAULT '{}',
    p_cluster_preferences jsonb DEFAULT '{}',
    p_prediction_accuracy float DEFAULT NULL,
    p_taste_vector float[] DEFAULT NULL
)
RETURNS uuid
LANGUAGE plpgsql
AS $$
DECLARE
    result_id uuid;
    taste_vec vector(512);
BEGIN
    -- Convert float array to vector if provided
    IF p_taste_vector IS NOT NULL AND array_length(p_taste_vector, 1) = 512 THEN
        taste_vec := p_taste_vector::vector(512);
    END IF;

    IF p_user_id IS NOT NULL THEN
        INSERT INTO user_seed_preferences (
            user_id, gender, rounds_completed, categories_tested,
            attribute_preferences, cluster_preferences, prediction_accuracy,
            taste_vector, completed_at, updated_at
        )
        VALUES (
            p_user_id, p_gender, p_rounds_completed, p_categories_tested,
            p_attribute_preferences, p_cluster_preferences, p_prediction_accuracy,
            taste_vec, NOW(), NOW()
        )
        ON CONFLICT (user_id) DO UPDATE SET
            gender = EXCLUDED.gender,
            rounds_completed = EXCLUDED.rounds_completed,
            categories_tested = EXCLUDED.categories_tested,
            attribute_preferences = EXCLUDED.attribute_preferences,
            cluster_preferences = EXCLUDED.cluster_preferences,
            prediction_accuracy = EXCLUDED.prediction_accuracy,
            taste_vector = EXCLUDED.taste_vector,
            completed_at = NOW(),
            updated_at = NOW()
        RETURNING id INTO result_id;
    ELSIF p_anon_id IS NOT NULL THEN
        INSERT INTO user_seed_preferences (
            anon_id, gender, rounds_completed, categories_tested,
            attribute_preferences, cluster_preferences, prediction_accuracy,
            taste_vector, completed_at, updated_at
        )
        VALUES (
            p_anon_id, p_gender, p_rounds_completed, p_categories_tested,
            p_attribute_preferences, p_cluster_preferences, p_prediction_accuracy,
            taste_vec, NOW(), NOW()
        )
        ON CONFLICT (anon_id) DO UPDATE SET
            gender = EXCLUDED.gender,
            rounds_completed = EXCLUDED.rounds_completed,
            categories_tested = EXCLUDED.categories_tested,
            attribute_preferences = EXCLUDED.attribute_preferences,
            cluster_preferences = EXCLUDED.cluster_preferences,
            prediction_accuracy = EXCLUDED.prediction_accuracy,
            taste_vector = EXCLUDED.taste_vector,
            completed_at = NOW(),
            updated_at = NOW()
        RETURNING id INTO result_id;
    ELSE
        RAISE EXCEPTION 'Either user_id or anon_id must be provided';
    END IF;

    RETURN result_id;
END;
$$;

-- =====================================================
-- STEP 8: Create updated_at trigger
-- =====================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply triggers
DROP TRIGGER IF EXISTS update_user_seed_preferences_updated_at ON user_seed_preferences;
CREATE TRIGGER update_user_seed_preferences_updated_at
    BEFORE UPDATE ON user_seed_preferences
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_user_vectors_updated_at ON user_vectors;
CREATE TRIGGER update_user_vectors_updated_at
    BEFORE UPDATE ON user_vectors
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_sku_popularity_updated_at ON sku_popularity;
CREATE TRIGGER update_sku_popularity_updated_at
    BEFORE UPDATE ON sku_popularity
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- Done! Verify tables were created:
-- =====================================================
-- SELECT table_name FROM information_schema.tables
-- WHERE table_schema = 'public'
-- AND table_name IN ('user_seed_preferences', 'user_vectors', 'sku_popularity');

-- Test match_embeddings function:
-- SELECT * FROM match_embeddings(
--     (SELECT embedding FROM image_embeddings LIMIT 1),
--     5
-- );
