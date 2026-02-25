-- =====================================================
-- Multimodal Embeddings: Combined FashionCLIP image + text embeddings
--
-- Each product gets a combined vector that encodes BOTH:
-- 1. Visual appearance (from FashionCLIP image encoder)
-- 2. Product attributes (from FashionCLIP text encoder on Gemini attributes)
--
-- combined = alpha * image_embedding + (1-alpha) * text_embedding
-- then L2-normalized to unit length.
--
-- This enables semantic search that matches descriptive terms like "ribbed"
-- or "quilted" that exist in product names/attributes but not in images.
--
-- Supports versioning for A/B testing:
--   v1 = structured attributes only (name, category, color, fabric, etc.)
--   v2 = structured attributes + source_description excerpt
-- =====================================================

-- Drop old versions if they exist (safe re-run)
DROP TABLE IF EXISTS product_multimodal_embeddings CASCADE;

CREATE TABLE product_multimodal_embeddings (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    product_id uuid NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    version smallint NOT NULL DEFAULT 1,
    image_embedding vector(512),           -- primary image embedding (for reference)
    text_embedding vector(512),            -- FashionCLIP text encoder on attributes
    multimodal_embedding vector(512) NOT NULL,  -- combined: alpha*image + (1-alpha)*text
    text_used text,                        -- the attribute string used (for debugging)
    alpha float NOT NULL DEFAULT 0.6,      -- image weight (text weight = 1 - alpha)
    created_at timestamptz DEFAULT now(),
    UNIQUE(product_id, version)
);

-- Index for fast lookups by product_id
CREATE INDEX idx_multimodal_product_id ON product_multimodal_embeddings (product_id);

-- Index for fast lookups by version (used in RPC WHERE clause)
CREATE INDEX idx_multimodal_version ON product_multimodal_embeddings (version);

-- =====================================================
-- search_multimodal: Search against combined embeddings
--
-- Key differences from text_search_products_filtered:
-- 1. JOINs product_multimodal_embeddings (1 row/product) instead of
--    image_embeddings (multiple rows/product) -- no DISTINCT ON needed
-- 2. Searches multimodal_embedding which encodes both visual + text
-- 3. Takes embedding_version parameter for A/B testing
-- 4. Simpler and faster due to fewer rows and no dedup
-- =====================================================

DROP FUNCTION IF EXISTS search_multimodal(vector, int, int, smallint, text[], text[], text[], numeric, numeric, uuid[]);

CREATE OR REPLACE FUNCTION search_multimodal(
    query_embedding vector(512),
    match_count int DEFAULT 50,
    match_offset int DEFAULT 0,
    embedding_version smallint DEFAULT 1,
    -- Filters (kept minimal -- soft scoring handles the rest in Python)
    filter_categories text[] DEFAULT NULL,
    exclude_brands text[] DEFAULT NULL,
    include_brands text[] DEFAULT NULL,
    min_price numeric DEFAULT NULL,
    max_price numeric DEFAULT NULL,
    exclude_product_ids uuid[] DEFAULT NULL
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
    original_price numeric,
    in_stock boolean,
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
    RETURN QUERY
    SELECT
        p.id as product_id,
        (1 - (pme.multimodal_embedding <=> query_embedding))::float as similarity,
        p.name,
        p.brand,
        p.category,
        p.broad_category,
        p.article_type,
        p.price,
        p.original_price,
        p.in_stock,
        p.primary_image_url,
        p.gallery_images,
        p.colors,
        p.materials,
        p.fit,
        p.length,
        p.sleeve
    FROM products p
    INNER JOIN product_multimodal_embeddings pme ON pme.product_id = p.id
    LEFT JOIN product_attributes pa ON pa.sku_id = p.id
    WHERE
        pme.version = embedding_version
        AND p.in_stock = true
        -- Category filter via Gemini category_l1 (broad_category is unpopulated)
        AND (filter_categories IS NULL OR LOWER(pa.category_l1) = ANY(filter_categories))
        -- Brand filters
        AND (exclude_brands IS NULL OR NOT (LOWER(p.brand) = ANY(SELECT LOWER(unnest(exclude_brands)))))
        AND (include_brands IS NULL OR LOWER(p.brand) = ANY(SELECT LOWER(unnest(include_brands))))
        -- Price range
        AND (min_price IS NULL OR p.price >= min_price)
        AND (max_price IS NULL OR p.price <= max_price)
        -- Exclude specific products
        AND (exclude_product_ids IS NULL OR NOT (p.id = ANY(exclude_product_ids)))
    ORDER BY pme.multimodal_embedding <=> query_embedding ASC
    LIMIT match_count
    OFFSET match_offset;
END;
$$;

-- Grant permissions
GRANT EXECUTE ON FUNCTION search_multimodal TO anon, authenticated, service_role;

-- RLS: allow authenticated users to read multimodal embeddings
ALTER TABLE product_multimodal_embeddings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow read access to multimodal embeddings"
    ON product_multimodal_embeddings FOR SELECT
    USING (true);

CREATE POLICY "Allow service role full access to multimodal embeddings"
    ON product_multimodal_embeddings FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);
