-- =====================================================================
-- Migration 047: Use Gemini category_l1 for category filtering
-- =====================================================================
-- Replaces p.category (DB field, ~10% error rate) with pa.category_l1
-- (Gemini-classified, accurate) in all three RPCs.
--
-- Changes:
--   1. LEFT JOIN product_attributes → INNER JOIN (only return Gemini-classified products)
--   2. p.category = filter_category → Gemini L1 broad mapping = filter_category
--
-- The CASE mapping mirrors Python _GEMINI_L1_TO_BROAD:
--   Tops/Top → tops, Bottoms/Bottom → bottoms, Dresses/Dress → dresses,
--   Outerwear/Jackets & Coats → outerwear, Activewear/Swimwear → tops,
--   Jumpsuits & Rompers → dresses
-- =====================================================================


-- Helper: map Gemini L1 to broad category (reusable across RPCs)
CREATE OR REPLACE FUNCTION gemini_l1_to_broad(l1 text)
RETURNS text
LANGUAGE sql
IMMUTABLE
AS $$
  SELECT CASE LOWER(TRIM(COALESCE(l1, '')))
    WHEN 'tops'                THEN 'tops'
    WHEN 'top'                 THEN 'tops'
    WHEN 'bottoms'             THEN 'bottoms'
    WHEN 'bottom'              THEN 'bottoms'
    WHEN 'dresses'             THEN 'dresses'
    WHEN 'dress'               THEN 'dresses'
    WHEN 'outerwear'           THEN 'outerwear'
    WHEN 'jackets & coats'     THEN 'outerwear'
    WHEN 'coats & jackets'     THEN 'outerwear'
    WHEN 'activewear'          THEN 'tops'
    WHEN 'swimwear'            THEN 'tops'
    WHEN 'jumpsuits & rompers' THEN 'dresses'
    WHEN 'jumpsuit'            THEN 'dresses'
    ELSE LOWER(TRIM(COALESCE(l1, '')))
  END;
$$;


-- =====================================================================
-- 1. UPDATED text_search_products
-- =====================================================================

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
  SET LOCAL hnsw.ef_search = 100;

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
    INNER JOIN product_attributes pa ON pa.sku_id = p.id
    WHERE
      ie.sku_id IS NOT NULL
      AND (filter_category IS NULL OR gemini_l1_to_broad(pa.category_l1) = filter_category)
    ORDER BY p.id, ie.embedding <=> query_embedding ASC
  ) AS unique_products
  ORDER BY similarity DESC
  LIMIT match_count
  OFFSET match_offset;
END;
$$;

SELECT 'text_search_products updated to use Gemini L1' AS status;


-- =====================================================================
-- 2. UPDATED get_similar_products_v2
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

  SELECT ie.embedding INTO source_embedding
  FROM image_embeddings ie
  WHERE ie.sku_id::uuid = source_product_id
  LIMIT 1;

  IF source_embedding IS NULL THEN
    RETURN;
  END IF;

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
    INNER JOIN product_attributes pa ON pa.sku_id = p.id
    WHERE
      ie.sku_id IS NOT NULL
      AND p.id != source_product_id
      AND (filter_category IS NULL OR gemini_l1_to_broad(pa.category_l1) = filter_category)
    ORDER BY p.id, ie.embedding <=> source_embedding ASC
  ) AS unique_products
  ORDER BY similarity DESC
  LIMIT match_count
  OFFSET match_offset;
END;
$$;

SELECT 'get_similar_products_v2 updated to use Gemini L1' AS status;


-- =====================================================================
-- 3. UPDATED batch_complement_search
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
  query_type text,
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
      INNER JOIN product_attributes pa ON pa.sku_id = p.id
      WHERE ie.sku_id IS NOT NULL
        AND p.id != source_product_id
        AND (filter_category IS NULL OR gemini_l1_to_broad(pa.category_l1) = filter_category)
      ORDER BY p.id, ie.embedding <=> source_embedding ASC
    ) sub
    ORDER BY similarity DESC
    LIMIT 60;
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
      INNER JOIN product_attributes pa ON pa.sku_id = p.id
      WHERE ie.sku_id IS NOT NULL
        AND p.id != source_product_id
        AND (filter_category IS NULL OR gemini_l1_to_broad(pa.category_l1) = filter_category)
      ORDER BY p.id, ie.embedding <=> prompt_embedding ASC
    ) sub
    ORDER BY similarity DESC
    LIMIT match_per_prompt;
  END LOOP;
END;
$$;

SELECT 'batch_complement_search updated to use Gemini L1' AS status;


-- =====================================================================
-- 4. GRANT PERMISSIONS
-- =====================================================================

GRANT EXECUTE ON FUNCTION gemini_l1_to_broad TO anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION text_search_products TO anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION get_similar_products_v2 TO anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION batch_complement_search TO anon, authenticated, service_role;

SELECT 'All permissions granted — migration 047 complete' AS status;
