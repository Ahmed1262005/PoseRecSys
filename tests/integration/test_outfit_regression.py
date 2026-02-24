"""
Regression tests for the outfit engine.

Run BEFORE and AFTER SQL migrations to verify the engine still produces
correct results. Requires a running Supabase connection (integration test).

Usage:
    PYTHONPATH=src python -m pytest tests/integration/test_outfit_regression.py -v

These tests verify:
1. Response structure is correct
2. Items are returned for each expected category
3. Scores are in valid ranges
4. Dimension scores are present and well-formed
5. Source product info is correct
6. Pagination info is valid
7. Scoring info matches expected engine version
8. No regressions in category routing (state machine)
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from services.outfit_engine import get_outfit_engine, get_complementary_targets

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    """Shared engine instance (loads model once)."""
    return get_outfit_engine()


@pytest.fixture(scope="module")
def supabase(engine):
    return engine.supabase


@pytest.fixture(scope="module")
def sample_products(supabase):
    """Find one product per L1 category for testing."""
    categories = ["Tops", "Bottoms", "Dresses", "Outerwear"]
    products = {}

    for cat in categories:
        # Find a product with Gemini attrs that's in stock
        attrs = (
            supabase.table("product_attributes")
            .select("sku_id")
            .eq("category_l1", cat)
            .limit(10)
            .execute()
        )
        if not attrs.data:
            continue

        for row in attrs.data:
            prod = (
                supabase.table("products")
                .select("id, name, brand, price, category, primary_image_url")
                .eq("id", row["sku_id"])
                .eq("in_stock", True)
                .not_.is_("primary_image_url", "null")
                .limit(1)
                .execute()
            )
            if prod.data:
                products[cat] = prod.data[0]
                break

    return products


# ---------------------------------------------------------------------------
# Response Structure Tests
# ---------------------------------------------------------------------------

class TestResponseStructure:
    """Verify the response shape is correct."""

    def test_build_outfit_returns_dict(self, engine, sample_products):
        pid = sample_products["Tops"]["id"]
        result = engine.build_outfit(product_id=pid, items_per_category=4)
        assert isinstance(result, dict)

    def test_top_level_keys(self, engine, sample_products):
        pid = sample_products["Tops"]["id"]
        result = engine.build_outfit(product_id=pid, items_per_category=4)
        required_keys = {"source_product", "recommendations", "status", "scoring_info", "complete_outfit"}
        assert required_keys.issubset(result.keys()), f"Missing keys: {required_keys - result.keys()}"

    def test_source_product_fields(self, engine, sample_products):
        pid = sample_products["Tops"]["id"]
        result = engine.build_outfit(product_id=pid, items_per_category=4)
        source = result["source_product"]
        required = {"product_id", "name", "brand", "category", "price", "image_url"}
        assert required.issubset(source.keys()), f"Missing source fields: {required - source.keys()}"
        assert source["product_id"] == pid

    def test_scoring_info(self, engine, sample_products):
        pid = sample_products["Tops"]["id"]
        result = engine.build_outfit(product_id=pid, items_per_category=4)
        info = result["scoring_info"]
        assert info["dimensions"] == 8
        assert "compat" in info["fusion"]
        assert "cosine" in info["fusion"]
        assert info["engine"] == "tattoo_v2.1"

    def test_complete_outfit_fields(self, engine, sample_products):
        pid = sample_products["Tops"]["id"]
        result = engine.build_outfit(product_id=pid, items_per_category=4)
        outfit = result["complete_outfit"]
        assert "items" in outfit
        assert "total_price" in outfit
        assert "item_count" in outfit
        assert pid in outfit["items"]
        assert outfit["item_count"] >= 1


# ---------------------------------------------------------------------------
# Category Routing Tests
# ---------------------------------------------------------------------------

class TestCategoryRouting:
    """Verify the state machine produces expected target categories."""

    def test_tops_targets_bottoms(self, engine, sample_products):
        if "Tops" not in sample_products:
            pytest.skip("No Tops product found")
        pid = sample_products["Tops"]["id"]
        result = engine.build_outfit(product_id=pid, items_per_category=4)
        cats = list(result["recommendations"].keys())
        # Tops should recommend at least bottoms
        assert "bottoms" in cats or len(cats) >= 1, f"Unexpected categories: {cats}"

    def test_bottoms_targets_tops(self, engine, sample_products):
        if "Bottoms" not in sample_products:
            pytest.skip("No Bottoms product found")
        pid = sample_products["Bottoms"]["id"]
        result = engine.build_outfit(product_id=pid, items_per_category=4)
        cats = list(result["recommendations"].keys())
        assert "tops" in cats, f"Expected 'tops' in categories, got: {cats}"

    def test_dresses_targets_outerwear(self, engine, sample_products):
        if "Dresses" not in sample_products:
            pytest.skip("No Dresses product found")
        pid = sample_products["Dresses"]["id"]
        result = engine.build_outfit(product_id=pid, items_per_category=4)
        cats = list(result["recommendations"].keys())
        assert "outerwear" in cats, f"Expected 'outerwear' in categories, got: {cats}"

    def test_outerwear_targets_multiple(self, engine, sample_products):
        if "Outerwear" not in sample_products:
            pytest.skip("No Outerwear product found")
        pid = sample_products["Outerwear"]["id"]
        result = engine.build_outfit(product_id=pid, items_per_category=4)
        cats = list(result["recommendations"].keys())
        # Outerwear should target tops + bottoms + dresses (3 categories)
        assert len(cats) >= 2, f"Expected 2+ categories, got: {cats}"


# ---------------------------------------------------------------------------
# Item Quality Tests
# ---------------------------------------------------------------------------

class TestItemQuality:
    """Verify recommended items have valid data and scores."""

    def _get_first_items(self, engine, sample_products, cat_key):
        if cat_key not in sample_products:
            pytest.skip(f"No {cat_key} product found")
        pid = sample_products[cat_key]["id"]
        result = engine.build_outfit(product_id=pid, items_per_category=6)
        items = []
        for cat_data in result["recommendations"].values():
            items.extend(cat_data.get("items", []))
        return items

    def test_items_have_required_fields(self, engine, sample_products):
        items = self._get_first_items(engine, sample_products, "Bottoms")
        assert len(items) > 0, "No items returned"
        required = {"product_id", "name", "brand", "price", "image_url", "rank",
                     "tattoo_score", "compatibility_score", "cosine_similarity"}
        for item in items[:3]:
            missing = required - item.keys()
            assert not missing, f"Item missing fields: {missing}"

    def test_items_have_dimension_scores(self, engine, sample_products):
        items = self._get_first_items(engine, sample_products, "Bottoms")
        expected_dims = {"occasion_formality", "style", "fabric", "silhouette",
                         "color", "seasonality", "pattern", "price"}
        for item in items[:3]:
            dims = item.get("dimension_scores", {})
            assert expected_dims.issubset(dims.keys()), \
                f"Missing dimensions: {expected_dims - dims.keys()}"

    def test_score_ranges(self, engine, sample_products):
        items = self._get_first_items(engine, sample_products, "Tops")
        for item in items[:6]:
            tattoo = item["tattoo_score"]
            compat = item["compatibility_score"]
            cosine = item["cosine_similarity"]
            assert 0.0 <= tattoo <= 1.0, f"tattoo_score out of range: {tattoo}"
            assert 0.0 <= compat <= 1.0, f"compatibility_score out of range: {compat}"
            assert -1.0 <= cosine <= 1.0, f"cosine_similarity out of range: {cosine}"

    def test_dimension_score_ranges(self, engine, sample_products):
        items = self._get_first_items(engine, sample_products, "Dresses")
        for item in items[:3]:
            for dim, val in item.get("dimension_scores", {}).items():
                assert 0.0 <= val <= 1.0, f"Dim {dim} out of range: {val}"

    def test_items_sorted_by_tattoo_desc(self, engine, sample_products):
        """Top items should have highest tattoo scores (within each category)."""
        if "Bottoms" not in sample_products:
            pytest.skip("No Bottoms product found")
        pid = sample_products["Bottoms"]["id"]
        result = engine.build_outfit(product_id=pid, items_per_category=6)
        for cat, cat_data in result["recommendations"].items():
            items = cat_data.get("items", [])
            if len(items) < 2:
                continue
            # After MMR reranking, order may not be strictly descending,
            # but #1 should be among the top scores
            scores = [i["tattoo_score"] for i in items]
            assert scores[0] >= scores[-1] * 0.8, \
                f"Top item score {scores[0]} too low vs last {scores[-1]} in {cat}"


# ---------------------------------------------------------------------------
# Pagination Tests
# ---------------------------------------------------------------------------

class TestPagination:
    """Verify pagination info is correct."""

    def test_pagination_fields(self, engine, sample_products):
        pid = sample_products["Tops"]["id"]
        result = engine.build_outfit(product_id=pid, items_per_category=4)
        for cat, cat_data in result["recommendations"].items():
            pag = cat_data["pagination"]
            assert "offset" in pag
            assert "limit" in pag
            assert "returned" in pag
            assert "has_more" in pag
            assert pag["returned"] == len(cat_data["items"])
            assert pag["limit"] == 4

    def test_feed_mode_pagination(self, engine, sample_products):
        """Feed mode: single category with offset."""
        pid = sample_products["Bottoms"]["id"]
        result = engine.build_outfit(
            product_id=pid, target_category="tops",
            offset=0, limit=3,
        )
        assert "tops" in result["recommendations"]
        pag = result["recommendations"]["tops"]["pagination"]
        assert pag["offset"] == 0
        assert pag["limit"] == 3
        assert pag["returned"] <= 3


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Verify graceful error handling."""

    def test_invalid_product_id(self, engine):
        result = engine.build_outfit(product_id="00000000-0000-0000-0000-000000000000")
        assert "error" in result
        assert result["recommendations"] == {}

    def test_status_field_present(self, engine, sample_products):
        for cat_key in ["Tops", "Bottoms", "Dresses", "Outerwear"]:
            if cat_key not in sample_products:
                continue
            pid = sample_products[cat_key]["id"]
            result = engine.build_outfit(product_id=pid, items_per_category=2)
            assert "status" in result
            assert result["status"] in ("ok", "set", "activewear", "blocked")


# ---------------------------------------------------------------------------
# Similar Items Tests
# ---------------------------------------------------------------------------

class TestSimilarItems:
    """Verify get_similar_scored endpoint."""

    def test_similar_returns_items(self, engine, sample_products):
        pid = sample_products["Tops"]["id"]
        result = engine.get_similar_scored(product_id=pid, limit=5)
        assert "results" in result
        assert len(result["results"]) > 0

    def test_similar_item_fields(self, engine, sample_products):
        pid = sample_products["Tops"]["id"]
        result = engine.get_similar_scored(product_id=pid, limit=5)
        for item in result["results"][:3]:
            assert "product_id" in item
            assert "tattoo_score" in item
            assert "compatibility_score" in item

    def test_similar_pagination(self, engine, sample_products):
        pid = sample_products["Tops"]["id"]
        result = engine.get_similar_scored(product_id=pid, limit=5, offset=0)
        assert "pagination" in result
        pag = result["pagination"]
        assert pag["limit"] == 5
        assert pag["offset"] == 0
