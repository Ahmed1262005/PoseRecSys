-- =====================================================
-- Fix Hybrid Search Ranking: Tiered Scoring System
-- Run this in Supabase Dashboard > SQL Editor
--
-- Problem: Brand/keyword boosts can override semantic relevance
-- - A 0.55 CLIP score + 0.38 brand boost = 0.93 beats a 0.85 CLIP score
-- - Exact product name searches fail (no prioritization)
--
-- Solution: Tiered scoring that preserves semantic relevance
-- - Tier 1: Exact name match (highest priority)
-- - Tier 2: Brand match with reduced weight (0.10 instead of 0.30)
-- - Tier 3: Name substring match (minimal boost: 0.03)
-- - Tier 4: Pure semantic score
-- =====================================================

-- Drop old versions if they exist
DROP FUNCTION IF EXISTS text_search_products_hybrid(
  vector(512), text, int, int, text[], text[], text[], text[], numeric, numeric, uuid[],
  text[], text[], text[], text[], text[], text[], text[], text[], text[], float, float, float, float
);

-- Create improved hybrid search function with tiered scoring
CREATE OR REPLACE FUNCTION text_search_products_hybrid(
  query_embedding vector(512),
  query_text text,                          -- Original query for keyword matching
  match_count int DEFAULT 50,
  match_offset int DEFAULT 0,
  -- Existing filters (same as text_search_products_filtered)
  filter_categories text[] DEFAULT NULL,
  exclude_colors text[] DEFAULT NULL,
  exclude_materials text[] DEFAULT NULL,
  exclude_brands text[] DEFAULT NULL,
  min_price numeric DEFAULT NULL,
  max_price numeric DEFAULT NULL,
  exclude_product_ids uuid[] DEFAULT NULL,
  -- Extended filters
  filter_article_types text[] DEFAULT NULL,
  include_colors text[] DEFAULT NULL,
  include_materials text[] DEFAULT NULL,
  include_brands text[] DEFAULT NULL,
  include_fits text[] DEFAULT NULL,
  include_occasions text[] DEFAULT NULL,
  exclude_styles text[] DEFAULT NULL,
  include_patterns text[] DEFAULT NULL,
  exclude_patterns text[] DEFAULT NULL,
  -- Thresholds
  occasion_threshold float DEFAULT 0.20,
  style_threshold float DEFAULT 0.25,
  -- Hybrid search weights (REDUCED from previous defaults)
  semantic_weight float DEFAULT 0.90,       -- Weight for CLIP similarity (was 0.7)
  keyword_weight float DEFAULT 0.10         -- Weight for keyword match boost (was 0.3)
)
RETURNS TABLE (
  product_id uuid,
  similarity float,                         -- Base CLIP similarity (raw score)
  keyword_match boolean,                    -- TRUE if query matched name/brand
  brand_match boolean,                      -- TRUE if query matched brand specifically
  exact_match boolean,                      -- TRUE if query exactly matches product name
  combined_score float,                     -- Final tiered score for ranking
  name text,
  brand text,
  category text,
  broad_category text,
  article_type text,
  price numeric,
  primary_image_url text,
  gallery_images text[],
  colors text[],
  materials text[],
  fit text,
  length text,
  sleeve text
)
LANGUAGE plpgsql
AS $$
DECLARE
  query_lower text;
  query_tokens text[];
BEGIN
  -- Normalize query for matching
  query_lower := LOWER(TRIM(query_text));

  -- Extract meaningful tokens (remove common words)
  query_tokens := array_remove(
    array_remove(
      array_remove(
        array_remove(
          array_remove(
            regexp_split_to_array(query_lower, '\s+'),
            'a'
          ),
          'the'
        ),
        'for'
      ),
      'with'
    ),
    'in'
  );

  -- Disable index scans for consistent ordering
  SET LOCAL enable_indexscan = off;
  SET LOCAL enable_bitmapscan = off;

  RETURN QUERY
  SELECT * FROM (
    SELECT DISTINCT ON (p.id)
      p.id as product_id,
      -- Raw CLIP similarity (preserved for transparency)
      (1 - (ie.embedding <=> query_embedding))::float as similarity,

      -- Check if any query token matches name or brand
      (
        p.name ILIKE '%' || query_lower || '%'
        OR p.brand ILIKE '%' || query_lower || '%'
        OR EXISTS (
          SELECT 1 FROM unnest(query_tokens) tok
          WHERE LENGTH(tok) >= 3  -- Only match tokens with 3+ chars
            AND (
              p.name ILIKE '%' || tok || '%'
              OR p.brand ILIKE '%' || tok || '%'
            )
        )
      ) as keyword_match,

      -- Check if brand specifically matches
      (
        p.brand ILIKE '%' || query_lower || '%'
        OR EXISTS (
          SELECT 1 FROM unnest(query_tokens) tok
          WHERE LENGTH(tok) >= 3
            AND p.brand ILIKE '%' || tok || '%'
        )
      ) as brand_match,

      -- NEW: Check for exact name match (case-insensitive)
      (
        LOWER(TRIM(p.name)) = query_lower
      ) as exact_match,

      -- TIERED SCORING SYSTEM
      -- Tier 1: Exact name match → 1.0 + base semantic (guaranteed top ranking)
      -- Tier 2: Brand match → semantic × 0.90 + keyword_weight (reduced boost)
      -- Tier 3: Name substring match → semantic × 0.95 + 0.03 (minimal boost)
      -- Tier 4: Pure semantic → semantic × 1.0 (no penalty)
      (
        CASE
          -- Tier 1: Exact name match (highest priority)
          WHEN LOWER(TRIM(p.name)) = query_lower THEN
            1.0 + (1 - (ie.embedding <=> query_embedding))::float

          -- Tier 2: Brand match (reduced weight)
          WHEN p.brand ILIKE '%' || query_lower || '%'
            OR EXISTS (
              SELECT 1 FROM unnest(query_tokens) tok
              WHERE LENGTH(tok) >= 3
                AND p.brand ILIKE '%' || tok || '%'
            )
          THEN
            (1 - (ie.embedding <=> query_embedding))::float * semantic_weight + keyword_weight

          -- Tier 3: Name substring match only (minimal boost, doesn't stack with brand)
          WHEN p.name ILIKE '%' || query_lower || '%'
            OR EXISTS (
              SELECT 1 FROM unnest(query_tokens) tok
              WHERE LENGTH(tok) >= 3 AND p.name ILIKE '%' || tok || '%'
            )
          THEN
            (1 - (ie.embedding <=> query_embedding))::float * 0.95 + 0.03

          -- Tier 4: Pure semantic (no boost, no penalty)
          ELSE
            (1 - (ie.embedding <=> query_embedding))::float
        END
      )::float as combined_score,

      p.name,
      p.brand,
      p.category,
      p.broad_category,
      p.article_type,
      p.price,
      p.primary_image_url,
      p.gallery_images,
      p.colors,
      p.materials,
      p.fit,
      p.length,
      p.sleeve
    FROM products p
    INNER JOIN image_embeddings ie ON ie.sku_id::uuid = p.id
    WHERE
      ie.sku_id IS NOT NULL
      AND p.in_stock = true
      -- Existing filters
      AND (filter_categories IS NULL OR p.broad_category = ANY(filter_categories))
      AND (filter_article_types IS NULL OR LOWER(p.article_type) = ANY(SELECT LOWER(unnest(filter_article_types))))
      AND (exclude_colors IS NULL OR NOT (p.colors && exclude_colors))
      AND (include_colors IS NULL OR p.colors && include_colors)
      AND (exclude_materials IS NULL OR NOT (p.materials && exclude_materials))
      AND (include_materials IS NULL OR p.materials && include_materials)
      AND (exclude_brands IS NULL OR NOT (LOWER(p.brand) = ANY(SELECT LOWER(unnest(exclude_brands)))))
      AND (include_brands IS NULL OR LOWER(p.brand) = ANY(SELECT LOWER(unnest(include_brands))))
      AND (min_price IS NULL OR p.price >= min_price)
      AND (max_price IS NULL OR p.price <= max_price)
      AND (exclude_product_ids IS NULL OR NOT (p.id = ANY(exclude_product_ids)))
      AND (include_fits IS NULL OR LOWER(p.fit) = ANY(SELECT LOWER(unnest(include_fits))))
      AND (include_occasions IS NULL OR matches_occasion(p.computed_occasion_scores, include_occasions, occasion_threshold))
      AND (exclude_styles IS NULL OR NOT has_excluded_style(p.computed_style_scores, exclude_styles, style_threshold))
      AND (include_patterns IS NULL OR p.style_tags && include_patterns)
      AND (exclude_patterns IS NULL OR NOT (p.style_tags && exclude_patterns))
    ORDER BY p.id, combined_score DESC
  ) AS unique_products
  ORDER BY combined_score DESC
  LIMIT match_count
  OFFSET match_offset;
END;
$$;

-- Grant permissions
GRANT EXECUTE ON FUNCTION text_search_products_hybrid TO anon, authenticated, service_role;

-- Verify creation
SELECT 'text_search_products_hybrid updated with tiered scoring' as status;

-- =====================================================
-- SCORING COMPARISON
-- =====================================================
--
-- BEFORE (broken):
-- | Query     | Item             | CLIP | Keyword | Brand Pref | Total |
-- |-----------|------------------|------|---------|------------|-------|
-- | "sweater" | Boohoo Crop Top  | 0.55 | +0.30   | +0.08      | 0.93  | ← WRONG: crop top beats sweater
-- | "sweater" | Generic Sweater  | 0.85 | 0       | 0          | 0.85  |
-- | "sweater" | Boohoo Sweater   | 0.75 | +0.30   | +0.08      | 1.13  |
--
-- AFTER (fixed - with new defaults semantic_weight=0.90, keyword_weight=0.10):
-- | Query     | Item             | CLIP | Tier | Calculation                    | Total |
-- |-----------|------------------|------|------|--------------------------------|-------|
-- | "sweater" | Boohoo Crop Top  | 0.55 | 4    | 0.55 (pure semantic)           | 0.55  | ← Now correctly low
-- | "sweater" | Generic Sweater  | 0.85 | 3    | 0.85 × 0.95 + 0.03 = 0.84      | 0.84  | ← Good
-- | "sweater" | Boohoo Sweater   | 0.75 | 2    | 0.75 × 0.90 + 0.10 = 0.775     | 0.78  | ← Brand boost helps but doesn't override
--
-- With user's Boohoo brand preference (+0.05 in Python):
-- | "sweater" | Boohoo Sweater   | 0.75 | 2    | 0.78 + 0.05 (capped) = 0.83    | 0.83  | ← Top rank (correct!)
--
-- Exact name match example:
-- | Query                                   | Item                                    | Total  |
-- |-----------------------------------------|-----------------------------------------|--------|
-- | "Harmony Balloon Sleeve Knit Sweater"   | Harmony Balloon Sleeve Knit Sweater     | 1.78+  | ← Exact match, guaranteed top
-- | "Harmony Balloon Sleeve Knit Sweater"   | Some Random Sweater                     | 0.72   | ← Semantic only
-- =====================================================
