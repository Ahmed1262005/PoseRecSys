-- Migration 049: Enable RLS on recommendation system tables
-- Fixes Supabase Security Advisor issues for tables owned by this project.
-- The full fix (including scraper tables) is in:
--   scraper/scripts/migrations/018_security_hardening.sql
--
-- Date: 2026-03-03

-- =============================================================================
-- user_seed_preferences: per-user taste vectors and Tinder test results
-- =============================================================================
ALTER TABLE public.user_seed_preferences ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
  CREATE POLICY "Users can view own seed preferences"
    ON public.user_seed_preferences FOR SELECT
    USING (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
  CREATE POLICY "Service role full access on user_seed_preferences"
    ON public.user_seed_preferences FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- =============================================================================
-- user_vectors: cached computed vectors for similarity search
-- =============================================================================
ALTER TABLE public.user_vectors ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
  CREATE POLICY "Users can view own vectors"
    ON public.user_vectors FOR SELECT
    USING (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
  CREATE POLICY "Service role full access on user_vectors"
    ON public.user_vectors FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- =============================================================================
-- sku_popularity: aggregate metrics (public read, service write)
-- =============================================================================
ALTER TABLE public.sku_popularity ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
  CREATE POLICY "Public read access on sku_popularity"
    ON public.sku_popularity FOR SELECT
    USING (true);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
  CREATE POLICY "Service role full access on sku_popularity"
    ON public.sku_popularity FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- =============================================================================
-- user_onboarding_profiles: per-user onboarding preferences
-- =============================================================================
ALTER TABLE public.user_onboarding_profiles ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
  CREATE POLICY "Users can view own onboarding profile"
    ON public.user_onboarding_profiles FOR SELECT
    USING (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
  CREATE POLICY "Users can update own onboarding profile"
    ON public.user_onboarding_profiles FOR UPDATE
    USING (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
  CREATE POLICY "Service role full access on user_onboarding_profiles"
    ON public.user_onboarding_profiles FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
