-- =====================================================
-- Hybrid Search: Semantic (CLIP) + Keyword Matching
-- Run this in Supabase Dashboard > SQL Editor
--
-- This function combines:
-- 1. CLIP vector similarity for semantic understanding
-- 2. Keyword matching on name/brand for direct queries
--
-- When user searches "Zara dress" or "reformation blouse",
-- items with matching brand/name get boosted in ranking.
-- =====================================================

-- Drop old versions if they exist
DROP FUNCTION IF EXISTS text_search_products_hybrid(
  vector(512), text, int, int, text[], text[], text[], text[], numeric, numeric, uuid[],
  text[], text[], text[], text[], text[], text[], text[], text[], text[], float, float, float, float
);

-- Create hybrid search function
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
  -- Hybrid search weights
  semantic_weight float DEFAULT 0.7,        -- Weight for CLIP similarity (0-1)
  keyword_weight float DEFAULT 0.3          -- Weight for keyword match boost (0-1)
)
RETURNS TABLE (
  product_id uuid,
  similarity float,                         -- Base CLIP similarity
  keyword_match boolean,                    -- TRUE if query matched name/brand
  brand_match boolean,                      -- TRUE if query matched brand specifically
  combined_score float,                     -- Weighted semantic + keyword score
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
      -- Check if brand specifically matches (higher priority)
      (
        p.brand ILIKE '%' || query_lower || '%'
        OR EXISTS (
          SELECT 1 FROM unnest(query_tokens) tok
          WHERE LENGTH(tok) >= 3
            AND p.brand ILIKE '%' || tok || '%'
        )
      ) as brand_match,
      -- Combined score: semantic + brand boost (full) + name boost (30%)
      (
        (1 - (ie.embedding <=> query_embedding))::float * semantic_weight
        -- Brand match gets full keyword_weight boost
        + CASE WHEN
            p.brand ILIKE '%' || query_lower || '%'
            OR EXISTS (
              SELECT 1 FROM unnest(query_tokens) tok
              WHERE LENGTH(tok) >= 3
                AND p.brand ILIKE '%' || tok || '%'
            )
          THEN keyword_weight
          ELSE 0
        END
        -- Name-only match gets 30% boost (doesn't stack with brand)
        + CASE WHEN
            NOT (p.brand ILIKE '%' || query_lower || '%')
            AND NOT EXISTS (
              SELECT 1 FROM unnest(query_tokens) tok
              WHERE LENGTH(tok) >= 3 AND p.brand ILIKE '%' || tok || '%'
            )
            AND (
              p.name ILIKE '%' || query_lower || '%'
              OR EXISTS (
                SELECT 1 FROM unnest(query_tokens) tok
                WHERE LENGTH(tok) >= 3 AND p.name ILIKE '%' || tok || '%'
              )
            )
          THEN keyword_weight * 0.3  -- Name match gets 30% of keyword boost
          ELSE 0
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
SELECT 'text_search_products_hybrid created successfully' as status;
