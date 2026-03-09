-- =====================================================================
-- Migration 073: Precomputed outfit candidate pools
-- =====================================================================
-- Stores top-60 nearest-neighbor candidates per source product per
-- target category, precomputed via faiss on exported FashionCLIP
-- embeddings.  At serve time the outfit engine does a simple indexed
-- lookup instead of live pgvector ANN search (~31s → <1s).
-- =====================================================================

CREATE TABLE IF NOT EXISTS outfit_candidates (
  source_id         uuid        NOT NULL,
  target_category   text        NOT NULL,   -- 'tops','bottoms','outerwear','dresses'
  candidate_id      uuid        NOT NULL,
  cosine_similarity float4      NOT NULL,
  rank              smallint    NOT NULL,    -- 1-based, 1 = most similar
  computed_at       timestamptz NOT NULL DEFAULT now(),

  PRIMARY KEY (source_id, target_category, rank)
);

-- Fast serve-time lookup: source + category → ordered candidates
CREATE INDEX IF NOT EXISTS idx_outfit_cand_lookup
  ON outfit_candidates (source_id, target_category, rank);

-- For staleness checks / incremental refresh
CREATE INDEX IF NOT EXISTS idx_outfit_cand_computed
  ON outfit_candidates (computed_at);

-- For finding which sources include a given candidate (cleanup on delete)
CREATE INDEX IF NOT EXISTS idx_outfit_cand_candidate
  ON outfit_candidates (candidate_id);


-- =====================================================================
-- RPC: Fetch precomputed candidates with joined product + attribute data
-- =====================================================================
-- Returns everything the scoring pipeline needs in a single call.

CREATE OR REPLACE FUNCTION get_outfit_candidates(
  p_source_id      uuid,
  p_target_category text,
  p_limit          int DEFAULT 60
)
RETURNS TABLE (
  candidate_id      uuid,
  cosine_similarity float,
  rank              smallint,
  name              text,
  brand             text,
  category          text,
  broad_category    text,
  price             numeric,
  primary_image_url text,
  gallery_images    text[],
  colors            text[],
  materials         text[],
  gemini_category_l1 text,
  gemini_category_l2 text,
  gemini_occasions   text[],
  gemini_style_tags  text[],
  gemini_pattern     text,
  gemini_formality   text,
  gemini_fit_type    text,
  gemini_color_family text,
  gemini_primary_color text,
  gemini_secondary_colors text[],
  gemini_seasons     text[],
  gemini_silhouette  text,
  gemini_construction jsonb,
  gemini_apparent_fabric text,
  gemini_texture     text,
  gemini_coverage_level text,
  gemini_sheen       text,
  gemini_rise        text,
  gemini_leg_shape   text,
  gemini_stretch     text,
  gemini_styling_metadata jsonb,
  gemini_styling_role text,
  gemini_appearance_top_tags text[],
  gemini_vibe_tags   text[],
  gemini_extractor_version text
)
LANGUAGE sql
STABLE
AS $$
  SELECT
    oc.candidate_id,
    oc.cosine_similarity::float,
    oc.rank,
    p.name,
    p.brand,
    p.category,
    p.broad_category,
    p.price,
    p.primary_image_url,
    p.gallery_images,
    p.colors,
    p.materials,
    pa.category_l1,
    pa.category_l2,
    pa.occasions,
    pa.style_tags,
    pa.pattern,
    pa.formality,
    pa.fit_type,
    pa.color_family,
    pa.primary_color,
    pa.secondary_colors,
    pa.seasons,
    pa.silhouette,
    pa.construction,
    pa.apparent_fabric,
    pa.texture,
    pa.coverage_level,
    pa.sheen,
    pa.rise,
    pa.leg_shape,
    pa.stretch,
    pa.styling_metadata,
    pa.styling_role,
    pa.appearance_top_tags,
    pa.vibe_tags,
    pa.extractor_version
  FROM outfit_candidates oc
  JOIN products p ON p.id = oc.candidate_id
  JOIN product_attributes pa ON pa.sku_id = oc.candidate_id
  WHERE oc.source_id = p_source_id
    AND oc.target_category = p_target_category
  ORDER BY oc.rank
  LIMIT p_limit;
$$;

GRANT SELECT ON outfit_candidates TO anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION get_outfit_candidates TO anon, authenticated, service_role;

SELECT 'Migration 073: outfit_candidates table + RPC created' AS status;
