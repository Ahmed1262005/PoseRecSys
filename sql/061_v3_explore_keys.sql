-- =============================================================================
-- V3 Feed: Precomputed Exploration Sort Keys
-- =============================================================================
--
-- Three deterministic pseudo-random sort keys per product. Each has a
-- B-tree index for efficient keyset pagination.
--
-- User assignment: key_family = hash(user_id) % 3 → picks 'a', 'b', or 'c'.
-- Different users see different shuffles. Same user sees consistent order.
--
-- Replaces the per-request md5(id::text || seed) computation that
-- forced full table scans.
--
-- RUN ORDER: 061 first, then 062, then 063.
-- =============================================================================

-- Add columns (idempotent)
ALTER TABLE products
    ADD COLUMN IF NOT EXISTS explore_key_a float8,
    ADD COLUMN IF NOT EXISTS explore_key_b float8,
    ADD COLUMN IF NOT EXISTS explore_key_c float8;

-- Populate with deterministic pseudo-random values in [0, 1)
-- Each key uses a different seed so the orderings are independent.
UPDATE products SET
    explore_key_a = (
        ('x' || substr(md5(id::text || 'v3_seed_alpha'), 1, 8))::bit(32)::bigint
        / 4294967295.0
    )::float8,
    explore_key_b = (
        ('x' || substr(md5(id::text || 'v3_seed_bravo'), 1, 8))::bit(32)::bigint
        / 4294967295.0
    )::float8,
    explore_key_c = (
        ('x' || substr(md5(id::text || 'v3_seed_charlie'), 1, 8))::bit(32)::bigint
        / 4294967295.0
    )::float8
WHERE explore_key_a IS NULL;   -- Only update rows not yet populated


-- B-tree indexes for keyset pagination (DESC for "highest first" ordering)
CREATE INDEX IF NOT EXISTS idx_products_explore_key_a
    ON products (explore_key_a DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_products_explore_key_b
    ON products (explore_key_b DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_products_explore_key_c
    ON products (explore_key_c DESC, id DESC);


-- =============================================================================
-- Verification query (run manually after migration)
-- =============================================================================
-- SELECT
--     count(*) AS total,
--     count(explore_key_a) AS has_key_a,
--     min(explore_key_a) AS min_a,
--     max(explore_key_a) AS max_a,
--     avg(explore_key_a) AS avg_a
-- FROM products
-- WHERE in_stock = true;
--
-- Expected: total ≈ 96K, has_key_a = total, min ≈ 0.0, max ≈ 1.0, avg ≈ 0.5
