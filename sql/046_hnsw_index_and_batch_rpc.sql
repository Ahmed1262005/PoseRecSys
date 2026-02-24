-- =====================================================================
-- Migration 046: HNSW index + batch search RPC + joined attributes
-- =====================================================================
-- This migration:
--   1. Creates an HNSW index on image_embeddings for fast ANN search
--   2. Updates text_search_products to USE the index (removes disable_indexscan)
--      and JOINs product_attributes to eliminate the separate enrichment call
--   3. Updates get_similar_products_v2 similarly
--   4. Creates batch_complement_search: accepts multiple embeddings in one
--      RPC call, returning results for all prompts at once
--
-- HNSW index parameters:
--   m=16          — connections per node (16 is good for 100K-500K vectors)
--   ef_construction=200 — build-time quality (higher = better recall, slower build)
--   Distance: cosine (vector_cosine_ops)
--
-- Run in Supabase SQL Editor. Index creation takes ~2-5 minutes on 170K vectors.
-- =====================================================================


-- =====================================================================
-- 1. HNSW INDEX
-- =====================================================================

-- Drop existing IVFFlat index if any (to avoid conflicts)
DROP INDEX IF EXISTS image_embeddings_embedding_idx;
DROP INDEX IF EXISTS image_embeddings_hnsw_idx;

-- Create HNSW index for cosine similarity
-- This replaces the brute-force sequential scan that was taking ~800ms per query
CREATE INDEX image_embeddings_hnsw_idx
  ON image_embeddings
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 200);

-- Analyze the table so the query planner uses the new index
ANALYZE image_embeddings;

SELECT 'HNSW index created on image_embeddings' AS status;


-- =====================================================================
-- 2. UPDATED text_search_products (uses HNSW + JOINs product_attributes)
-- =====================================================================

DROP FUNCTION IF EXISTS text_search_products(vector, int, text);
DROP FUNCTION IF EXISTS text_search_products(vector, int, int, text);

CREATE OR REPLACE FUNCTION text_search_products(
  query_embedding vector(512),
  match_count int DEFAULT 50,
  match_offset int DEFAULT 0,
  filter_category text DEFAULT NULL
)
RETURNS TABLE (
  product_id uuid,
  similarity float,
  name text,
  brand text,
  category text,
  broad_category text,
  price numeric,
  primary_image_url text,
  gallery_images text[],
  colors text[],
  materials text[],
  -- Gemini attributes (joined)
  gemini_category_l1 text,
  gemini_category_l2 text,
  gemini_occasions text[],
  gemini_style_tags text[],
  gemini_pattern text,
  gemini_formality text,
  gemini_fit_type text,
  gemini_color_family text,
  gemini_primary_color text,
  gemini_secondary_colors text[],
  gemini_seasons text[],
  gemini_silhouette text,
  gemini_construction jsonb,
  gemini_apparent_fabric text,
  gemini_texture text,
  gemini_coverage_level text,
  gemini_sheen text,
  gemini_rise text,
  gemini_leg_shape text,
  gemini_stretch text
)
LANGUAGE plpgsql
AS $$
BEGIN
  -- Set HNSW search quality (higher = better recall, slightly slower)
  SET LOCAL hnsw.ef_search = 100;

  -- Deduplicated results with joined Gemini attributes
  RETURN QUERY
  SELECT * FROM (
    SELECT DISTINCT ON (p.id)
      p.id AS product_id,
      (1 - (ie.embedding <=> query_embedding))::float AS similarity,
      p.name,
      p.brand,
      p.category,
      p.broad_category,
      p.price,
      p.primary_image_url,
      p.gallery_images,
      p.colors,
      p.materials,
      -- Gemini attributes
      pa.category_l1  AS gemini_category_l1,
      pa.category_l2  AS gemini_category_l2,
      pa.occasions     AS gemini_occasions,
      pa.style_tags    AS gemini_style_tags,
      pa.pattern       AS gemini_pattern,
      pa.formality     AS gemini_formality,
      pa.fit_type      AS gemini_fit_type,
      pa.color_family  AS gemini_color_family,
      pa.primary_color AS gemini_primary_color,
      pa.secondary_colors AS gemini_secondary_colors,
      pa.seasons       AS gemini_seasons,
      pa.silhouette    AS gemini_silhouette,
      pa.construction  AS gemini_construction,
      pa.apparent_fabric AS gemini_apparent_fabric,
      pa.texture       AS gemini_texture,
      pa.coverage_level AS gemini_coverage_level,
      pa.sheen         AS gemini_sheen,
      pa.rise          AS gemini_rise,
      pa.leg_shape     AS gemini_leg_shape,
      pa.stretch       AS gemini_stretch
    FROM products p
    INNER JOIN image_embeddings ie ON ie.sku_id::uuid = p.id
    LEFT JOIN product_attributes pa ON pa.sku_id = p.id
    WHERE
      ie.sku_id IS NOT NULL
      AND (filter_category IS NULL OR p.category = filter_category)
    ORDER BY p.id, ie.embedding <=> query_embedding ASC
  ) AS unique_products
  ORDER BY similarity DESC
  LIMIT match_count
  OFFSET match_offset;
END;
$$;

SELECT 'text_search_products updated with HNSW + joined attributes' AS status;


-- =====================================================================
-- 3. UPDATED get_similar_products_v2 (uses HNSW + JOINs product_attributes)
-- =====================================================================

DROP FUNCTION IF EXISTS get_similar_products_v2(uuid, int, int, text);

CREATE OR REPLACE FUNCTION get_similar_products_v2(
  source_product_id uuid,
  match_count int DEFAULT 50,
  match_offset int DEFAULT 0,
  filter_category text DEFAULT NULL
)
RETURNS TABLE (
  product_id uuid,
  similarity float,
  name text,
  brand text,
  category text,
  broad_category text,
  price numeric,
  primary_image_url text,
  gallery_images text[],
  colors text[],
  materials text[],
  -- Gemini attributes (joined)
  gemini_category_l1 text,
  gemini_category_l2 text,
  gemini_occasions text[],
  gemini_style_tags text[],
  gemini_pattern text,
  gemini_formality text,
  gemini_fit_type text,
  gemini_color_family text,
  gemini_primary_color text,
  gemini_secondary_colors text[],
  gemini_seasons text[],
  gemini_silhouette text,
  gemini_construction jsonb,
  gemini_apparent_fabric text,
  gemini_texture text,
  gemini_coverage_level text,
  gemini_sheen text,
  gemini_rise text,
  gemini_leg_shape text,
  gemini_stretch text
)
LANGUAGE plpgsql
AS $$
DECLARE
  source_embedding vector(512);
BEGIN
  SET LOCAL hnsw.ef_search = 100;

  -- Get the embedding for the source product
  SELECT ie.embedding INTO source_embedding
  FROM image_embeddings ie
  WHERE ie.sku_id::uuid = source_product_id
  LIMIT 1;

  IF source_embedding IS NULL THEN
    RETURN;
  END IF;

  -- Deduplicated results with joined Gemini attributes
  RETURN QUERY
  SELECT * FROM (
    SELECT DISTINCT ON (p.id)
      p.id AS product_id,
      (1 - (ie.embedding <=> source_embedding))::float AS similarity,
      p.name,
      p.brand,
      p.category,
      p.broad_category,
      p.price,
      p.primary_image_url,
      p.gallery_images,
      p.colors,
      p.materials,
      pa.category_l1  AS gemini_category_l1,
      pa.category_l2  AS gemini_category_l2,
      pa.occasions     AS gemini_occasions,
      pa.style_tags    AS gemini_style_tags,
      pa.pattern       AS gemini_pattern,
      pa.formality     AS gemini_formality,
      pa.fit_type      AS gemini_fit_type,
      pa.color_family  AS gemini_color_family,
      pa.primary_color AS gemini_primary_color,
      pa.secondary_colors AS gemini_secondary_colors,
      pa.seasons       AS gemini_seasons,
      pa.silhouette    AS gemini_silhouette,
      pa.construction  AS gemini_construction,
      pa.apparent_fabric AS gemini_apparent_fabric,
      pa.texture       AS gemini_texture,
      pa.coverage_level AS gemini_coverage_level,
      pa.sheen         AS gemini_sheen,
      pa.rise          AS gemini_rise,
      pa.leg_shape     AS gemini_leg_shape,
      pa.stretch       AS gemini_stretch
    FROM products p
    INNER JOIN image_embeddings ie ON ie.sku_id::uuid = p.id
    LEFT JOIN product_attributes pa ON pa.sku_id = p.id
    WHERE
      ie.sku_id IS NOT NULL
      AND p.id != source_product_id
      AND (filter_category IS NULL OR p.category = filter_category)
    ORDER BY p.id, ie.embedding <=> source_embedding ASC
  ) AS unique_products
  ORDER BY similarity DESC
  LIMIT match_count
  OFFSET match_offset;
END;
$$;

SELECT 'get_similar_products_v2 updated with HNSW + joined attributes' AS status;


-- =====================================================================
-- 4. NEW batch_complement_search (multi-embedding in one call)
-- =====================================================================
-- Accepts a JSONB array of embedding vectors + a pgvector source embedding.
-- Returns results tagged with query_index so the caller knows which
-- embedding produced each result.
--
-- Usage from Python:
--   supabase.rpc("batch_complement_search", {
--       "source_product_id": "...",
--       "prompt_embeddings_json": "[[0.1,0.2,...], [0.3,0.4,...], ...]",
--       "match_per_prompt": 8,
--       "filter_category": "tops"
--   })
-- =====================================================================

DROP FUNCTION IF EXISTS batch_complement_search(uuid, jsonb, int, text);

CREATE OR REPLACE FUNCTION batch_complement_search(
  source_product_id uuid,
  prompt_embeddings_json jsonb,
  match_per_prompt int DEFAULT 8,
  filter_category text DEFAULT NULL
)
RETURNS TABLE (
  query_index int,
  query_type text,          -- 'pgvector' or 'prompt_N'
  product_id uuid,
  similarity float,
  name text,
  brand text,
  category text,
  broad_category text,
  price numeric,
  primary_image_url text,
  gallery_images text[],
  colors text[],
  materials text[],
  -- Gemini attributes
  gemini_category_l1 text,
  gemini_category_l2 text,
  gemini_occasions text[],
  gemini_style_tags text[],
  gemini_pattern text,
  gemini_formality text,
  gemini_fit_type text,
  gemini_color_family text,
  gemini_primary_color text,
  gemini_secondary_colors text[],
  gemini_seasons text[],
  gemini_silhouette text,
  gemini_construction jsonb,
  gemini_apparent_fabric text,
  gemini_texture text,
  gemini_coverage_level text,
  gemini_sheen text,
  gemini_rise text,
  gemini_leg_shape text,
  gemini_stretch text
)
LANGUAGE plpgsql
AS $$
DECLARE
  source_embedding vector(512);
  prompt_embedding vector(512);
  prompt_array jsonb;
  i int;
  num_prompts int;
BEGIN
  SET LOCAL hnsw.ef_search = 100;

  -- Get source product embedding for pgvector pool
  SELECT ie.embedding INTO source_embedding
  FROM image_embeddings ie
  WHERE ie.sku_id::uuid = source_product_id
  LIMIT 1;

  -- Pool 1: product-to-product similarity (query_index = 0, query_type = 'pgvector')
  IF source_embedding IS NOT NULL THEN
    RETURN QUERY
    SELECT * FROM (
      SELECT DISTINCT ON (p.id)
        0 AS query_index,
        'pgvector'::text AS query_type,
        p.id AS product_id,
        (1 - (ie.embedding <=> source_embedding))::float AS similarity,
        p.name, p.brand, p.category, p.broad_category, p.price,
        p.primary_image_url, p.gallery_images, p.colors, p.materials,
        pa.category_l1, pa.category_l2, pa.occasions, pa.style_tags,
        pa.pattern, pa.formality, pa.fit_type, pa.color_family,
        pa.primary_color, pa.secondary_colors, pa.seasons, pa.silhouette,
        pa.construction, pa.apparent_fabric, pa.texture, pa.coverage_level,
        pa.sheen, pa.rise, pa.leg_shape, pa.stretch
      FROM products p
      INNER JOIN image_embeddings ie ON ie.sku_id::uuid = p.id
      LEFT JOIN product_attributes pa ON pa.sku_id = p.id
      WHERE ie.sku_id IS NOT NULL
        AND p.id != source_product_id
        AND (filter_category IS NULL OR p.category = filter_category)
      ORDER BY p.id, ie.embedding <=> source_embedding ASC
    ) sub
    ORDER BY similarity DESC
    LIMIT 60;  -- pgvector pool size
  END IF;

  -- Pool 2: text prompt embeddings (query_index = 1..N, query_type = 'prompt_N')
  num_prompts := jsonb_array_length(prompt_embeddings_json);
  FOR i IN 0 .. num_prompts - 1 LOOP
    prompt_array := prompt_embeddings_json -> i;
    prompt_embedding := (
      SELECT array_agg(val::float)::vector(512)
      FROM jsonb_array_elements_text(prompt_array) AS val
    );

    RETURN QUERY
    SELECT * FROM (
      SELECT DISTINCT ON (p.id)
        (i + 1) AS query_index,
        ('prompt_' || i)::text AS query_type,
        p.id AS product_id,
        (1 - (ie.embedding <=> prompt_embedding))::float AS similarity,
        p.name, p.brand, p.category, p.broad_category, p.price,
        p.primary_image_url, p.gallery_images, p.colors, p.materials,
        pa.category_l1, pa.category_l2, pa.occasions, pa.style_tags,
        pa.pattern, pa.formality, pa.fit_type, pa.color_family,
        pa.primary_color, pa.secondary_colors, pa.seasons, pa.silhouette,
        pa.construction, pa.apparent_fabric, pa.texture, pa.coverage_level,
        pa.sheen, pa.rise, pa.leg_shape, pa.stretch
      FROM products p
      INNER JOIN image_embeddings ie ON ie.sku_id::uuid = p.id
      LEFT JOIN product_attributes pa ON pa.sku_id = p.id
      WHERE ie.sku_id IS NOT NULL
        AND p.id != source_product_id
        AND (filter_category IS NULL OR p.category = filter_category)
      ORDER BY p.id, ie.embedding <=> prompt_embedding ASC
    ) sub
    ORDER BY similarity DESC
    LIMIT match_per_prompt;
  END LOOP;
END;
$$;

SELECT 'batch_complement_search created' AS status;


-- =====================================================================
-- 5. GRANT PERMISSIONS
-- =====================================================================

GRANT EXECUTE ON FUNCTION text_search_products TO anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION get_similar_products_v2 TO anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION batch_complement_search TO anon, authenticated, service_role;

SELECT 'All permissions granted' AS status;
