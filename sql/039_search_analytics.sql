-- =====================================================
-- Search Analytics Tables
-- Run this in Supabase Dashboard > SQL Editor
--
-- Tracks search queries, clicks, and conversions
-- for search quality analysis and improvement.
-- =====================================================

-- Search queries log
CREATE TABLE IF NOT EXISTS search_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query TEXT NOT NULL,
    query_normalized TEXT NOT NULL,
    user_id UUID,
    session_id TEXT,
    intent TEXT,  -- 'exact', 'specific', 'vague'
    total_results INT DEFAULT 0,
    algolia_results INT DEFAULT 0,
    semantic_results INT DEFAULT 0,
    filters JSONB DEFAULT '{}',
    latency_ms INT DEFAULT 0,
    algolia_latency_ms INT DEFAULT 0,
    semantic_latency_ms INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for analytics queries
CREATE INDEX IF NOT EXISTS idx_search_analytics_query
    ON search_analytics(query_normalized);
CREATE INDEX IF NOT EXISTS idx_search_analytics_created
    ON search_analytics(created_at);
CREATE INDEX IF NOT EXISTS idx_search_analytics_user
    ON search_analytics(user_id);
CREATE INDEX IF NOT EXISTS idx_search_analytics_intent
    ON search_analytics(intent);

-- Search clicks tracking
CREATE TABLE IF NOT EXISTS search_clicks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query TEXT NOT NULL,
    product_id UUID,
    position INT,
    user_id UUID,
    session_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_search_clicks_query
    ON search_clicks(query);
CREATE INDEX IF NOT EXISTS idx_search_clicks_product
    ON search_clicks(product_id);
CREATE INDEX IF NOT EXISTS idx_search_clicks_created
    ON search_clicks(created_at);

-- Search conversions (add-to-cart / purchase from search)
CREATE TABLE IF NOT EXISTS search_conversions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query TEXT NOT NULL,
    product_id UUID,
    user_id UUID,
    session_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_search_conversions_query
    ON search_conversions(query);
CREATE INDEX IF NOT EXISTS idx_search_conversions_created
    ON search_conversions(created_at);

-- Grant permissions
GRANT ALL ON search_analytics TO anon, authenticated, service_role;
GRANT ALL ON search_clicks TO anon, authenticated, service_role;
GRANT ALL ON search_conversions TO anon, authenticated, service_role;

-- Enable RLS (optional - allow service_role full access, authenticated users read own)
ALTER TABLE search_analytics ENABLE ROW LEVEL SECURITY;
ALTER TABLE search_clicks ENABLE ROW LEVEL SECURITY;
ALTER TABLE search_conversions ENABLE ROW LEVEL SECURITY;

-- Service role can do everything
CREATE POLICY search_analytics_service ON search_analytics
    FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY search_clicks_service ON search_clicks
    FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY search_conversions_service ON search_conversions
    FOR ALL TO service_role USING (true) WITH CHECK (true);

-- Authenticated users can insert (their own events)
CREATE POLICY search_analytics_insert ON search_analytics
    FOR INSERT TO authenticated WITH CHECK (true);
CREATE POLICY search_clicks_insert ON search_clicks
    FOR INSERT TO authenticated WITH CHECK (true);
CREATE POLICY search_conversions_insert ON search_conversions
    FOR INSERT TO authenticated WITH CHECK (true);

SELECT 'search_analytics tables created' as status;
