-- NEW search function for women's fashion text search
-- Creates NEW functions (doesn't replace existing ones)
-- Apply this in Supabase SQL Editor (Database > SQL Editor)

-- Drop if exists (signature changed for pagination)
DROP FUNCTION IF EXISTS text_search_products(vector, int, text);
DROP FUNCTION IF EXISTS text_search_products(vector, int, int, text);
DROP FUNCTION IF EXISTS text_search_products_filtered(vector, int, text[], text[], text[], text[], numeric, numeric, uuid[]);
DROP FUNCTION IF EXISTS text_search_products_filtered(vector, int, int, text[], text[], text[], text[], numeric, numeric, uuid[]);
DROP FUNCTION IF EXISTS text_search_products_filtered(vector, int, int, text[], text[], text[], text[], numeric, numeric, uuid[], text[], text[], text[], text[], text[], text[], text[], text[], text[], float, float);

-- =====================================================
-- text_search_products: Full-featured text search with pagination
-- Returns all product details needed by the frontend
-- NO similarity threshold - returns up to match_count results
-- =====================================================
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
  materials text[]
)
LANGUAGE plpgsql
AS $$
BEGIN
  -- Force exact search (bypass index for full recall)
  SET LOCAL enable_indexscan = off;
  SET LOCAL enable_bitmapscan = off;

  -- Deduplicated results - one per product (best matching image)
  RETURN QUERY
  SELECT * FROM (
    SELECT DISTINCT ON (p.id)
      p.id as product_id,
      (1 - (ie.embedding <=> query_embedding))::float as similarity,
      p.name,
      p.brand,
      p.category,
      p.broad_category,
      p.price,
      p.primary_image_url,
      p.gallery_images,
      p.colors,
      p.materials
    FROM products p
    INNER JOIN image_embeddings ie ON ie.sku_id::uuid = p.id
    WHERE
      -- Only products with embeddings
      ie.sku_id IS NOT NULL
      -- Category filter (optional)
      AND (filter_category IS NULL OR p.category = filter_category)
    ORDER BY p.id, ie.embedding <=> query_embedding ASC
  ) AS unique_products
  ORDER BY similarity DESC
  LIMIT match_count
  OFFSET match_offset;
END;
$$;

-- =====================================================
-- text_search_products_filtered: With hard filters + pagination
-- Extended with article_type, include filters, occasions, styles, patterns
-- =====================================================
CREATE OR REPLACE FUNCTION text_search_products_filtered(
  query_embedding vector(512),
  match_count int DEFAULT 50,
  match_offset int DEFAULT 0,
  -- Existing filters
  filter_categories text[] DEFAULT NULL,
  exclude_colors text[] DEFAULT NULL,
  exclude_materials text[] DEFAULT NULL,
  exclude_brands text[] DEFAULT NULL,
  min_price numeric DEFAULT NULL,
  max_price numeric DEFAULT NULL,
  exclude_product_ids uuid[] DEFAULT NULL,
  -- NEW filters
  filter_article_types text[] DEFAULT NULL,
  include_colors text[] DEFAULT NULL,
  include_materials text[] DEFAULT NULL,
  include_brands text[] DEFAULT NULL,
  include_fits text[] DEFAULT NULL,
  include_occasions text[] DEFAULT NULL,
  exclude_styles text[] DEFAULT NULL,
  include_patterns text[] DEFAULT NULL,
  exclude_patterns text[] DEFAULT NULL,
  occasion_threshold float DEFAULT 0.20,
  style_threshold float DEFAULT 0.25
)
RETURNS TABLE (
  product_id uuid,
  similarity float,
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
BEGIN
  -- Force exact search (bypass index for full recall)
  SET LOCAL enable_indexscan = off;
  SET LOCAL enable_bitmapscan = off;

  -- Deduplicated results - one per product (best matching image)
  RETURN QUERY
  SELECT * FROM (
    SELECT DISTINCT ON (p.id)
      p.id as product_id,
      (1 - (ie.embedding <=> query_embedding))::float as similarity,
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
      -- Category filter (array) - matches broad_category
      AND (filter_categories IS NULL OR p.broad_category = ANY(filter_categories))
      -- Article type filter (specific types like "jeans", "t-shirts")
      AND (filter_article_types IS NULL OR LOWER(p.article_type) = ANY(SELECT LOWER(unnest(filter_article_types))))
      -- Exclude colors (use array overlap operator)
      AND (exclude_colors IS NULL OR NOT (p.colors && exclude_colors))
      -- Include colors (positive filter)
      AND (include_colors IS NULL OR p.colors && include_colors)
      -- Exclude materials
      AND (exclude_materials IS NULL OR NOT (p.materials && exclude_materials))
      -- Include materials (positive filter)
      AND (include_materials IS NULL OR p.materials && include_materials)
      -- Exclude brands (case-insensitive)
      AND (exclude_brands IS NULL OR NOT (LOWER(p.brand) = ANY(SELECT LOWER(unnest(exclude_brands)))))
      -- Include brands (positive filter, case-insensitive)
      AND (include_brands IS NULL OR LOWER(p.brand) = ANY(SELECT LOWER(unnest(include_brands))))
      -- Price range
      AND (min_price IS NULL OR p.price >= min_price)
      AND (max_price IS NULL OR p.price <= max_price)
      -- Exclude specific products
      AND (exclude_product_ids IS NULL OR NOT (p.id = ANY(exclude_product_ids)))
      -- Fit filter (case-insensitive)
      AND (include_fits IS NULL OR LOWER(p.fit) = ANY(SELECT LOWER(unnest(include_fits))))
      -- Occasion filter (uses computed_occasion_scores)
      AND (include_occasions IS NULL OR matches_occasion(p.computed_occasion_scores, include_occasions, occasion_threshold))
      -- Style exclusion filter (uses computed_style_scores)
      AND (exclude_styles IS NULL OR NOT has_excluded_style(p.computed_style_scores, exclude_styles, style_threshold))
      -- Pattern filters (using style_tags which contains pattern info)
      AND (include_patterns IS NULL OR p.style_tags && include_patterns)
      AND (exclude_patterns IS NULL OR NOT (p.style_tags && exclude_patterns))
    ORDER BY p.id, ie.embedding <=> query_embedding ASC
  ) AS unique_products
  ORDER BY similarity DESC
  LIMIT match_count
  OFFSET match_offset;
END;
$$;

-- =====================================================
-- get_similar_products_v2: Find similar products with full recall
-- =====================================================
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
  materials text[]
)
LANGUAGE plpgsql
AS $$
DECLARE
  source_embedding vector(512);
BEGIN
  -- Force exact search for full recall
  SET LOCAL enable_indexscan = off;
  SET LOCAL enable_bitmapscan = off;

  -- Get the embedding for the source product
  SELECT ie.embedding INTO source_embedding
  FROM image_embeddings ie
  WHERE ie.sku_id::uuid = source_product_id
  LIMIT 1;

  -- If no embedding found, return empty
  IF source_embedding IS NULL THEN
    RETURN;
  END IF;

  -- Find similar products (deduplicated - one result per product)
  RETURN QUERY
  SELECT * FROM (
    SELECT DISTINCT ON (p.id)
      p.id as product_id,
      (1 - (ie.embedding <=> source_embedding))::float as similarity,
      p.name,
      p.brand,
      p.category,
      p.broad_category,
      p.price,
      p.primary_image_url,
      p.gallery_images,
      p.colors,
      p.materials
    FROM products p
    INNER JOIN image_embeddings ie ON ie.sku_id::uuid = p.id
    WHERE
      ie.sku_id IS NOT NULL
      AND p.id != source_product_id  -- Exclude source product
      AND (filter_category IS NULL OR p.category = filter_category)
    ORDER BY p.id, ie.embedding <=> source_embedding ASC
  ) AS unique_products
  ORDER BY similarity DESC
  LIMIT match_count
  OFFSET match_offset;
END;
$$;

-- =====================================================
-- get_complementary_products: Find products from complementary categories
-- Returns full product details needed for scoring
-- =====================================================
DROP FUNCTION IF EXISTS get_complementary_products(uuid, text[], int);

CREATE OR REPLACE FUNCTION get_complementary_products(
  source_product_id uuid,
  target_categories text[],
  match_count int DEFAULT 20
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
  base_color text,
  materials text[],
  occasions text[],
  usage text
)
LANGUAGE plpgsql
AS $$
DECLARE
  source_embedding vector(512);
BEGIN
  -- Force exact search for full recall
  SET LOCAL enable_indexscan = off;
  SET LOCAL enable_bitmapscan = off;

  -- Get the embedding for the source product
  SELECT ie.embedding INTO source_embedding
  FROM image_embeddings ie
  WHERE ie.sku_id::uuid = source_product_id
  LIMIT 1;

  -- If no embedding found, return empty
  IF source_embedding IS NULL THEN
    RETURN;
  END IF;

  -- Find complementary products (deduplicated)
  RETURN QUERY
  SELECT * FROM (
    SELECT DISTINCT ON (p.id)
      p.id as product_id,
      (1 - (ie.embedding <=> source_embedding))::float as similarity,
      p.name,
      p.brand,
      p.category,
      p.broad_category,
      p.price,
      p.primary_image_url,
      p.gallery_images,
      p.colors,
      p.base_color,
      p.materials,
      p.occasions,
      p.usage
    FROM products p
    INNER JOIN image_embeddings ie ON ie.sku_id::uuid = p.id
    WHERE
      ie.sku_id IS NOT NULL
      AND p.id != source_product_id
      AND p.category = ANY(target_categories)
    ORDER BY p.id, ie.embedding <=> source_embedding ASC
  ) AS unique_products
  ORDER BY similarity DESC
  LIMIT match_count;
END;
$$;

-- =====================================================
-- get_product_details: Get full details of a single product
-- =====================================================
DROP FUNCTION IF EXISTS get_product_details(uuid);

CREATE OR REPLACE FUNCTION get_product_details(
  product_id_param uuid
)
RETURNS TABLE (
  product_id uuid,
  name text,
  brand text,
  category text,
  broad_category text,
  price numeric,
  primary_image_url text,
  gallery_images text[],
  colors text[],
  base_color text,
  materials text[],
  occasions text[],
  usage text
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
    p.broad_category,
    p.price,
    p.primary_image_url,
    p.gallery_images,
    p.colors,
    p.base_color,
    p.materials,
    p.occasions,
    p.usage
  FROM products p
  WHERE p.id = product_id_param
  LIMIT 1;
END;
$$;

-- Grant execute permissions
GRANT EXECUTE ON FUNCTION text_search_products TO anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION text_search_products_filtered TO anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION get_similar_products_v2 TO anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION get_complementary_products TO anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION get_product_details TO anon, authenticated, service_role;
