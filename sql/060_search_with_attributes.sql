-- ============================================================
-- search_semantic_with_attributes
--
-- pgvector semantic search + v1.0.0.2 attribute filtering.
-- Extends search_multimodal with fine-grained visual filters.
--
-- When NO attribute filters are set, behaves exactly like
-- search_multimodal (LEFT JOIN, all products eligible).
-- When attribute filters ARE set, only products that have
-- matching v1.0.0.2 attributes pass through.
-- ============================================================

-- Drop ALL existing overloads so CREATE OR REPLACE doesn't conflict.
-- CASCADE drops dependent objects (none expected).
DROP FUNCTION IF EXISTS search_semantic_with_attributes CASCADE;

CREATE OR REPLACE FUNCTION search_semantic_with_attributes(
    query_embedding vector(512),
    match_count int DEFAULT 60,
    match_offset int DEFAULT 0,
    embedding_version smallint DEFAULT 1,

    -- Basic filters (same as search_multimodal)
    filter_category_l1 text[] DEFAULT NULL,
    filter_category_l2 text[] DEFAULT NULL,
    filter_brands text[] DEFAULT NULL,
    exclude_brands text[] DEFAULT NULL,
    filter_min_price numeric DEFAULT NULL,
    filter_max_price numeric DEFAULT NULL,
    exclude_product_ids uuid[] DEFAULT NULL,

    -- Coverage (Layer 4A)
    filter_arm_coverage text[] DEFAULT NULL,
    filter_shoulder_coverage text[] DEFAULT NULL,
    filter_neckline_depth text[] DEFAULT NULL,
    filter_midriff_exposure text[] DEFAULT NULL,
    filter_back_openness text[] DEFAULT NULL,
    filter_sheerness text[] DEFAULT NULL,

    -- Shape / silhouette (Layer 4E)
    filter_body_cling text[] DEFAULT NULL,
    filter_structure_level text[] DEFAULT NULL,
    filter_drape_level text[] DEFAULT NULL,
    filter_cropped_degree text[] DEFAULT NULL,
    filter_waist_definition text[] DEFAULT NULL,
    filter_leg_volume text[] DEFAULT NULL,
    filter_bulk text[] DEFAULT NULL,

    -- Details (Layer 4 + 5)
    filter_has_pockets boolean DEFAULT NULL,
    filter_pocket_types text[] DEFAULT NULL,
    filter_pocket_has_zip boolean DEFAULT NULL,
    filter_slit_presence boolean DEFAULT NULL,
    filter_slit_height text[] DEFAULT NULL,
    filter_detail_tags text[] DEFAULT NULL,
    filter_lining text[] DEFAULT NULL,

    -- Occasions / style
    filter_occasions text[] DEFAULT NULL,
    filter_formality text[] DEFAULT NULL,
    filter_seasons text[] DEFAULT NULL
)
RETURNS TABLE (
    product_id uuid,
    similarity float,
    -- Product fields
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
    -- Attribute fields (NULL for products without v1.0.0.2 data)
    pa_category_l1 text,
    pa_category_l2 text,
    pa_body_cling text,
    pa_structure_level text,
    pa_drape_level text,
    pa_arm_coverage text,
    pa_shoulder_coverage text,
    pa_neckline_depth text,
    pa_midriff_exposure text,
    pa_back_openness text,
    pa_sheerness text,
    pa_cropped_degree text,
    pa_waist_definition text,
    pa_has_pockets boolean,
    pa_pocket_details jsonb,
    pa_slit_presence boolean,
    pa_slit_height text,
    pa_detail_tags text[],
    pa_vibe_tags text[],
    pa_appearance_tags text[],
    pa_occasions text[],
    pa_formality text,
    pa_style_tags text[]
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id as product_id,
        (1 - (pme.multimodal_embedding <=> query_embedding))::float as similarity,
        -- Product
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
        -- Attributes
        pa.category_l1      as pa_category_l1,
        pa.category_l2      as pa_category_l2,
        pa.body_cling_visual as pa_body_cling,
        pa.structure_level   as pa_structure_level,
        pa.drape_level       as pa_drape_level,
        pa.arm_coverage      as pa_arm_coverage,
        pa.shoulder_coverage as pa_shoulder_coverage,
        pa.neckline_depth    as pa_neckline_depth,
        pa.midriff_exposure  as pa_midriff_exposure,
        pa.back_openness     as pa_back_openness,
        pa.sheerness_visual  as pa_sheerness,
        pa.cropped_degree    as pa_cropped_degree,
        pa.waist_definition_visual as pa_waist_definition,
        pa.has_pockets_visible     as pa_has_pockets,
        pa.pocket_details          as pa_pocket_details,
        pa.slit_presence           as pa_slit_presence,
        pa.slit_height             as pa_slit_height,
        pa.detail_tags             as pa_detail_tags,
        pa.vibe_tags               as pa_vibe_tags,
        pa.appearance_top_tags     as pa_appearance_tags,
        pa.occasions               as pa_occasions,
        pa.formality               as pa_formality,
        pa.style_tags              as pa_style_tags
    FROM products p
    INNER JOIN product_multimodal_embeddings pme ON pme.product_id = p.id
    LEFT JOIN product_attributes pa ON pa.sku_id = p.id
    WHERE
        pme.version = embedding_version
        AND p.in_stock = true

        -- Basic filters
        AND (filter_category_l1 IS NULL OR pa.category_l1 = ANY(filter_category_l1))
        AND (filter_category_l2 IS NULL OR pa.category_l2 = ANY(filter_category_l2))
        AND (filter_brands IS NULL OR p.brand = ANY(filter_brands))
        AND (exclude_brands IS NULL
             OR NOT (LOWER(p.brand) = ANY(SELECT LOWER(unnest(exclude_brands)))))
        AND (filter_min_price IS NULL OR p.price >= filter_min_price)
        AND (filter_max_price IS NULL OR p.price <= filter_max_price)
        AND (exclude_product_ids IS NULL OR NOT (p.id = ANY(exclude_product_ids)))

        -- Coverage filters (Layer 4A)
        AND (filter_arm_coverage IS NULL      OR pa.arm_coverage = ANY(filter_arm_coverage))
        AND (filter_shoulder_coverage IS NULL  OR pa.shoulder_coverage = ANY(filter_shoulder_coverage))
        AND (filter_neckline_depth IS NULL     OR pa.neckline_depth = ANY(filter_neckline_depth))
        AND (filter_midriff_exposure IS NULL   OR pa.midriff_exposure = ANY(filter_midriff_exposure))
        AND (filter_back_openness IS NULL      OR pa.back_openness = ANY(filter_back_openness))
        AND (filter_sheerness IS NULL          OR pa.sheerness_visual = ANY(filter_sheerness))

        -- Shape filters (Layer 4E)
        AND (filter_body_cling IS NULL         OR pa.body_cling_visual = ANY(filter_body_cling))
        AND (filter_structure_level IS NULL     OR pa.structure_level = ANY(filter_structure_level))
        AND (filter_drape_level IS NULL         OR pa.drape_level = ANY(filter_drape_level))
        AND (filter_cropped_degree IS NULL      OR pa.cropped_degree = ANY(filter_cropped_degree))
        AND (filter_waist_definition IS NULL    OR pa.waist_definition_visual = ANY(filter_waist_definition))
        AND (filter_leg_volume IS NULL          OR pa.leg_volume_visual = ANY(filter_leg_volume))
        AND (filter_bulk IS NULL                OR pa.bulk_visual = ANY(filter_bulk))

        -- Detail filters (Layer 4 + 5)
        AND (filter_has_pockets IS NULL    OR pa.has_pockets_visible = filter_has_pockets)
        -- Pocket type filter: case-insensitive match against pocket_details.types JSONB array
        AND (filter_pocket_types IS NULL
             OR EXISTS (
                 SELECT 1
                 FROM jsonb_array_elements_text(COALESCE(pa.pocket_details->'types', '[]'::jsonb)) elem
                 WHERE LOWER(elem) = ANY(filter_pocket_types)
             ))
        -- Zippered pocket filter: zip_count > 0 OR any type containing "zip"
        AND (filter_pocket_has_zip IS NULL
             OR filter_pocket_has_zip = false
             OR (
                 pa.pocket_details IS NOT NULL
                 AND (
                     COALESCE((pa.pocket_details->>'zip_count')::int, 0) > 0
                     OR EXISTS (
                         SELECT 1
                         FROM jsonb_array_elements_text(COALESCE(pa.pocket_details->'types', '[]'::jsonb)) elem
                         WHERE LOWER(elem) LIKE '%zip%'
                     )
                 )
             ))
        AND (filter_slit_presence IS NULL  OR pa.slit_presence = filter_slit_presence)
        AND (filter_slit_height IS NULL    OR pa.slit_height = ANY(filter_slit_height))
        AND (filter_detail_tags IS NULL    OR pa.detail_tags @> filter_detail_tags)
        AND (filter_lining IS NULL         OR pa.lining_status_likely = ANY(filter_lining))

        -- Occasion / style filters
        AND (filter_occasions IS NULL  OR pa.occasions && filter_occasions)
        AND (filter_formality IS NULL  OR pa.formality = ANY(filter_formality))
        AND (filter_seasons IS NULL    OR pa.seasons && filter_seasons)

    ORDER BY pme.multimodal_embedding <=> query_embedding ASC
    LIMIT match_count
    OFFSET match_offset;
END;
$$;

-- Grant permissions
GRANT EXECUTE ON FUNCTION search_semantic_with_attributes TO anon, authenticated, service_role;
