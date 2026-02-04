-- Update text_search_products_filtered with extended filters
-- Run this in Supabase Dashboard > SQL Editor

-- Drop old versions
DROP FUNCTION IF EXISTS text_search_products_filtered(vector, int, text[], text[], text[], text[], numeric, numeric, uuid[]);
DROP FUNCTION IF EXISTS text_search_products_filtered(vector, int, int, text[], text[], text[], text[], numeric, numeric, uuid[]);
DROP FUNCTION IF EXISTS text_search_products_filtered(vector, int, int, text[], text[], text[], text[], numeric, numeric, uuid[], text[], text[], text[], text[], text[], text[], text[], text[], text[], float, float);

-- Create updated function with all new filters
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
  SET LOCAL enable_indexscan = off;
  SET LOCAL enable_bitmapscan = off;

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
    ORDER BY p.id, ie.embedding <=> query_embedding ASC
  ) AS unique_products
  ORDER BY similarity DESC
  LIMIT match_count
  OFFSET match_offset;
END;
$$;

-- Grant permissions
GRANT EXECUTE ON FUNCTION text_search_products_filtered TO anon, authenticated, service_role;

-- Verify
SELECT 'text_search_products_filtered updated successfully' as status;
