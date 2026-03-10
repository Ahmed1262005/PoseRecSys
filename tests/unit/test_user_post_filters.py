"""
Unit tests for the user UI filter / planner filter separation architecture.

Tests cover:
1. _snapshot_user_filters — captures user-set filters, ignores defaults
2. _strip_filters_from_request — creates pipeline_request without user filters
3. _apply_user_post_filters — brand, price, color, category, attribute, exclusion
4. No-op base case — no user filters means pipeline_request == request
5. Mixed user + planner scenario — the "athleisure + Nike" bug fix

Run with: PYTHONPATH=src python -m pytest tests/unit/test_user_post_filters.py -v
"""

import pytest
from typing import Any, Dict, List, Optional


# =============================================================================
# Helpers
# =============================================================================

def _make_request(**kwargs) -> "HybridSearchRequest":
    """Create a HybridSearchRequest with sensible defaults."""
    from search.models import HybridSearchRequest
    defaults = {"query": "test query"}
    defaults.update(kwargs)
    return HybridSearchRequest(**defaults)


def _make_result(
    product_id: str,
    brand: str = "TestBrand",
    price: float = 50.0,
    rrf_score: float = 0.5,
    **kwargs,
) -> dict:
    """Create a mock search result dict."""
    base = {
        "product_id": product_id,
        "name": f"Product {product_id}",
        "brand": brand,
        "image_url": f"https://img.example.com/{product_id}.jpg",
        "gallery_images": [],
        "price": price,
        "original_price": None,
        "is_on_sale": False,
        "is_set": False,
        "category_l1": "Tops",
        "category_l2": "T-Shirt",
        "broad_category": "tops",
        "article_type": "t-shirt",
        "primary_color": "Black",
        "color_family": "Dark",
        "pattern": "Solid",
        "apparent_fabric": "Cotton",
        "fit_type": "Regular",
        "formality": "Casual",
        "silhouette": "Straight",
        "length": "Regular",
        "neckline": "Crew",
        "sleeve_type": "Short",
        "rise": None,
        "style_tags": ["casual"],
        "occasions": ["everyday"],
        "seasons": ["spring", "summer"],
        "colors": ["black"],
        "materials": ["cotton"],
        "trending_score": 0,
        "source": "algolia",
        "rrf_score": rrf_score,
    }
    base.update(kwargs)
    return base


def _service():
    """Create a HybridSearchService for calling static/class methods."""
    from search.hybrid_search import HybridSearchService
    return HybridSearchService


# =============================================================================
# 1. _snapshot_user_filters
# =============================================================================

class TestSnapshotUserFilters:
    """Verify that _snapshot_user_filters captures the right fields."""

    def test_empty_request_returns_empty_snapshot(self):
        """No user-set filters → empty dict."""
        svc = _service()
        req = _make_request()
        snap = svc._snapshot_user_filters(req)
        assert snap == {}

    def test_brand_filter_captured(self):
        """User-set brands should appear in the snapshot."""
        svc = _service()
        req = _make_request(brands=["Nike", "Adidas"])
        snap = svc._snapshot_user_filters(req)
        assert snap["brands"] == ["Nike", "Adidas"]

    def test_price_filters_captured(self):
        """min_price and max_price should appear when set."""
        svc = _service()
        req = _make_request(min_price=20.0, max_price=100.0)
        snap = svc._snapshot_user_filters(req)
        assert snap["min_price"] == 20.0
        assert snap["max_price"] == 100.0

    def test_on_sale_only_false_not_captured(self):
        """on_sale_only defaults to False — should NOT appear."""
        svc = _service()
        req = _make_request(on_sale_only=False)
        snap = svc._snapshot_user_filters(req)
        assert "on_sale_only" not in snap

    def test_on_sale_only_true_captured(self):
        """on_sale_only=True is user-set."""
        svc = _service()
        req = _make_request(on_sale_only=True)
        snap = svc._snapshot_user_filters(req)
        assert snap["on_sale_only"] is True

    def test_is_set_captured_when_true(self):
        """is_set=True should be captured."""
        svc = _service()
        req = _make_request(is_set=True)
        snap = svc._snapshot_user_filters(req)
        assert snap["is_set"] is True

    def test_is_set_captured_when_false(self):
        """is_set=False should be captured (user explicitly excluding sets)."""
        svc = _service()
        req = _make_request(is_set=False)
        snap = svc._snapshot_user_filters(req)
        assert snap["is_set"] is False

    def test_colors_captured(self):
        svc = _service()
        req = _make_request(colors=["Red", "Blue"])
        snap = svc._snapshot_user_filters(req)
        assert snap["colors"] == ["Red", "Blue"]

    def test_category_l1_captured(self):
        svc = _service()
        req = _make_request(category_l1=["Dresses"])
        snap = svc._snapshot_user_filters(req)
        assert snap["category_l1"] == ["Dresses"]

    def test_exclude_brands_captured(self):
        svc = _service()
        req = _make_request(exclude_brands=["Shein"])
        snap = svc._snapshot_user_filters(req)
        assert snap["exclude_brands"] == ["Shein"]

    def test_non_filter_fields_excluded(self):
        """Control fields like query, page, sort_by must never appear."""
        svc = _service()
        req = _make_request(
            query="summer dress",
            page=3,
            page_size=25,
            sort_by="price_asc",
            session_id="abc",
            semantic_boost=0.6,
        )
        snap = svc._snapshot_user_filters(req)
        for field in ("query", "page", "page_size", "sort_by",
                      "session_id", "semantic_boost"):
            assert field not in snap

    def test_multiple_filters_captured(self):
        """Multiple user filters in one request."""
        svc = _service()
        req = _make_request(
            brands=["Nike"],
            colors=["White"],
            min_price=30.0,
            formality=["Casual"],
        )
        snap = svc._snapshot_user_filters(req)
        assert set(snap.keys()) == {"brands", "colors", "min_price", "formality"}

    def test_exclusion_filters_captured(self):
        """exclude_* fields should be in the snapshot."""
        svc = _service()
        req = _make_request(
            exclude_neckline=["Strapless"],
            exclude_sleeve_type=["Sleeveless"],
        )
        snap = svc._snapshot_user_filters(req)
        assert snap["exclude_neckline"] == ["Strapless"]
        assert snap["exclude_sleeve_type"] == ["Sleeveless"]

    def test_attribute_filters_captured(self):
        """All attribute filter types should be capturable."""
        svc = _service()
        req = _make_request(
            patterns=["Floral"],
            materials=["Silk"],
            occasions=["Date Night"],
            seasons=["Summer"],
            fit_type=["Fitted"],
            neckline=["V-Neck"],
            sleeve_type=["Long"],
            length=["Midi"],
            rise=["High"],
            silhouette=["A-Line"],
            article_type=["Dress"],
            style_tags=["bohemian"],
        )
        snap = svc._snapshot_user_filters(req)
        expected_keys = {
            "patterns", "materials", "occasions", "seasons",
            "fit_type", "neckline", "sleeve_type", "length",
            "rise", "silhouette", "article_type", "style_tags",
        }
        assert set(snap.keys()) == expected_keys


# =============================================================================
# 2. _strip_filters_from_request
# =============================================================================

class TestStripFiltersFromRequest:
    """Verify that _strip_filters_from_request removes the right fields."""

    def test_empty_filter_set_returns_same_request(self):
        """No filters to strip → same request object."""
        svc = _service()
        req = _make_request(brands=["Nike"])
        result = svc._strip_filters_from_request(req, set())
        assert result is req  # identity — no copy needed

    def test_brands_stripped(self):
        svc = _service()
        req = _make_request(brands=["Nike"], colors=["Red"])
        result = svc._strip_filters_from_request(req, {"brands"})
        assert result.brands is None
        assert result.colors == ["Red"]  # untouched

    def test_price_stripped(self):
        svc = _service()
        req = _make_request(min_price=20.0, max_price=100.0)
        result = svc._strip_filters_from_request(req, {"min_price", "max_price"})
        assert result.min_price is None
        assert result.max_price is None

    def test_on_sale_only_stripped_to_false(self):
        """on_sale_only should reset to False, not None."""
        svc = _service()
        req = _make_request(on_sale_only=True)
        result = svc._strip_filters_from_request(req, {"on_sale_only"})
        assert result.on_sale_only is False

    def test_is_set_stripped_to_none(self):
        svc = _service()
        req = _make_request(is_set=True)
        result = svc._strip_filters_from_request(req, {"is_set"})
        assert result.is_set is None

    def test_multiple_filters_stripped(self):
        svc = _service()
        req = _make_request(
            brands=["Nike"],
            colors=["Blue"],
            min_price=10.0,
            formality=["Casual"],
        )
        result = svc._strip_filters_from_request(
            req, {"brands", "colors", "min_price", "formality"},
        )
        assert result.brands is None
        assert result.colors is None
        assert result.min_price is None
        assert result.formality is None
        # Query survives
        assert result.query == "test query"

    def test_query_never_stripped(self):
        """Even if accidentally passed, query should survive."""
        svc = _service()
        req = _make_request()
        # query is not a filter, but test robustness
        result = svc._strip_filters_from_request(req, {"brands"})
        assert result.query == "test query"

    def test_planner_filters_survive_stripping(self):
        """If request has both user and planner fields, only user fields are stripped."""
        svc = _service()
        req = _make_request(
            brands=["Nike"],         # user-set
            category_l1=["Tops"],    # planner-set
            exclude_neckline=["Strapless"],  # planner-set
        )
        # Only strip brands (user-set)
        result = svc._strip_filters_from_request(req, {"brands"})
        assert result.brands is None
        assert result.category_l1 == ["Tops"]
        assert result.exclude_neckline == ["Strapless"]


# =============================================================================
# 3. _apply_user_post_filters — brand filtering
# =============================================================================

class TestUserPostFilterBrands:
    """Verify brand filtering on merged results."""

    def test_brand_inclusion_filter(self):
        """Only products matching the user's brand survive."""
        svc = _service()()
        results = [
            _make_result("1", brand="Nike"),
            _make_result("2", brand="Adidas"),
            _make_result("3", brand="Nike"),
            _make_result("4", brand="Puma"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"brands": ["Nike"]},
        )
        assert len(filtered) == 2
        assert all(r["brand"] == "Nike" for r in filtered)

    def test_brand_inclusion_case_insensitive(self):
        """Brand matching should be case-insensitive."""
        svc = _service()()
        results = [
            _make_result("1", brand="NIKE"),
            _make_result("2", brand="nike"),
            _make_result("3", brand="Nike"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"brands": ["nike"]},
        )
        assert len(filtered) == 3

    def test_brand_exclusion_filter(self):
        """Products matching excluded brands are dropped."""
        svc = _service()()
        results = [
            _make_result("1", brand="Nike"),
            _make_result("2", brand="Shein"),
            _make_result("3", brand="Zara"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"exclude_brands": ["Shein"]},
        )
        assert len(filtered) == 2
        assert all(r["brand"] != "Shein" for r in filtered)

    def test_brand_with_missing_brand_field(self):
        """Products with no brand field are excluded when brand filter is set."""
        svc = _service()()
        results = [
            _make_result("1", brand="Nike"),
            {"product_id": "2", "price": 30.0, "rrf_score": 0.3},
        ]
        filtered = svc._apply_user_post_filters(
            results, {"brands": ["Nike"]},
        )
        assert len(filtered) == 1
        assert filtered[0]["product_id"] == "1"

    def test_multiple_brands(self):
        """User selects multiple brands."""
        svc = _service()()
        results = [
            _make_result("1", brand="Nike"),
            _make_result("2", brand="Adidas"),
            _make_result("3", brand="Puma"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"brands": ["Nike", "Adidas"]},
        )
        assert len(filtered) == 2
        brands = {r["brand"] for r in filtered}
        assert brands == {"Nike", "Adidas"}


# =============================================================================
# 4. _apply_user_post_filters — price filtering
# =============================================================================

class TestUserPostFilterPrice:
    """Verify price filtering on merged results."""

    def test_min_price_filter(self):
        svc = _service()()
        results = [
            _make_result("1", price=10.0),
            _make_result("2", price=30.0),
            _make_result("3", price=50.0),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"min_price": 25.0},
        )
        assert len(filtered) == 2
        assert all(r["price"] >= 25.0 for r in filtered)

    def test_max_price_filter(self):
        svc = _service()()
        results = [
            _make_result("1", price=10.0),
            _make_result("2", price=30.0),
            _make_result("3", price=150.0),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"max_price": 50.0},
        )
        assert len(filtered) == 2
        assert all(r["price"] <= 50.0 for r in filtered)

    def test_price_range_filter(self):
        svc = _service()()
        results = [
            _make_result("1", price=10.0),
            _make_result("2", price=30.0),
            _make_result("3", price=80.0),
            _make_result("4", price=150.0),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"min_price": 20.0, "max_price": 100.0},
        )
        assert len(filtered) == 2
        assert {r["product_id"] for r in filtered} == {"2", "3"}

    def test_price_filter_skips_null_prices(self):
        """Products with None price are excluded when price filter is set."""
        svc = _service()()
        results = [
            _make_result("1", price=30.0),
            {**_make_result("2"), "price": None},
        ]
        filtered = svc._apply_user_post_filters(
            results, {"min_price": 10.0},
        )
        assert len(filtered) == 1
        assert filtered[0]["product_id"] == "1"


# =============================================================================
# 5. _apply_user_post_filters — color filtering
# =============================================================================

class TestUserPostFilterColors:
    """Verify color filtering on merged results."""

    def test_color_matches_primary_color(self):
        svc = _service()()
        results = [
            _make_result("1", primary_color="Red"),
            _make_result("2", primary_color="Blue"),
            _make_result("3", primary_color="Red"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"colors": ["Red"]},
        )
        assert len(filtered) == 2

    def test_color_matches_colors_list(self):
        """Should also check the 'colors' list field."""
        svc = _service()()
        results = [
            _make_result("1", primary_color="Black", colors=["black", "red"]),
            _make_result("2", primary_color="Blue", colors=["blue"]),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"colors": ["Red"]},
        )
        # Product 1 has "red" in its colors list
        assert len(filtered) == 1
        assert filtered[0]["product_id"] == "1"

    def test_color_case_insensitive(self):
        svc = _service()()
        results = [
            _make_result("1", primary_color="RED"),
            _make_result("2", primary_color="blue"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"colors": ["red"]},
        )
        assert len(filtered) == 1

    def test_color_family_filter(self):
        svc = _service()()
        results = [
            _make_result("1", color_family="Dark"),
            _make_result("2", color_family="Pastel"),
            _make_result("3", color_family="Dark"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"color_family": ["Dark"]},
        )
        assert len(filtered) == 2


# =============================================================================
# 6. _apply_user_post_filters — category filtering
# =============================================================================

class TestUserPostFilterCategories:
    """Verify category filtering on merged results."""

    def test_broad_category_filter(self):
        svc = _service()()
        results = [
            _make_result("1", broad_category="tops"),
            _make_result("2", broad_category="dresses"),
            _make_result("3", broad_category="tops"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"categories": ["tops"]},
        )
        assert len(filtered) == 2

    def test_category_l1_filter(self):
        svc = _service()()
        results = [
            _make_result("1", category_l1="Tops"),
            _make_result("2", category_l1="Dresses"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"category_l1": ["Dresses"]},
        )
        assert len(filtered) == 1
        assert filtered[0]["category_l1"] == "Dresses"

    def test_category_l2_filter(self):
        svc = _service()()
        results = [
            _make_result("1", category_l2="T-Shirt"),
            _make_result("2", category_l2="Blouse"),
            _make_result("3", category_l2="T-Shirt"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"category_l2": ["T-Shirt"]},
        )
        assert len(filtered) == 2


# =============================================================================
# 7. _apply_user_post_filters — single-value attribute filters
# =============================================================================

class TestUserPostFilterSingleAttrs:
    """Verify single-value attribute filtering (pattern, formality, etc.)."""

    def test_pattern_filter(self):
        svc = _service()()
        results = [
            _make_result("1", pattern="Floral"),
            _make_result("2", pattern="Solid"),
            _make_result("3", pattern="Floral"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"patterns": ["Floral"]},
        )
        assert len(filtered) == 2

    def test_formality_filter(self):
        svc = _service()()
        results = [
            _make_result("1", formality="Casual"),
            _make_result("2", formality="Formal"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"formality": ["Formal"]},
        )
        assert len(filtered) == 1

    def test_neckline_filter(self):
        svc = _service()()
        results = [
            _make_result("1", neckline="V-Neck"),
            _make_result("2", neckline="Crew"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"neckline": ["V-Neck"]},
        )
        assert len(filtered) == 1

    def test_fit_type_filter(self):
        svc = _service()()
        results = [
            _make_result("1", fit_type="Fitted"),
            _make_result("2", fit_type="Oversized"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"fit_type": ["Fitted"]},
        )
        assert len(filtered) == 1

    def test_sleeve_type_filter(self):
        svc = _service()()
        results = [
            _make_result("1", sleeve_type="Long"),
            _make_result("2", sleeve_type="Short"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"sleeve_type": ["Long"]},
        )
        assert len(filtered) == 1

    def test_length_filter(self):
        svc = _service()()
        results = [
            _make_result("1", length="Midi"),
            _make_result("2", length="Mini"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"length": ["Midi"]},
        )
        assert len(filtered) == 1

    def test_silhouette_filter(self):
        svc = _service()()
        results = [
            _make_result("1", silhouette="A-Line"),
            _make_result("2", silhouette="Straight"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"silhouette": ["A-Line"]},
        )
        assert len(filtered) == 1

    def test_article_type_filter(self):
        svc = _service()()
        results = [
            _make_result("1", article_type="dress"),
            _make_result("2", article_type="t-shirt"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"article_type": ["dress"]},
        )
        assert len(filtered) == 1

    def test_rise_filter(self):
        svc = _service()()
        results = [
            _make_result("1", rise="High"),
            _make_result("2", rise="Mid"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"rise": ["High"]},
        )
        assert len(filtered) == 1

    def test_multiple_values_accepted(self):
        """User picks e.g. formality=['Casual', 'Smart Casual']."""
        svc = _service()()
        results = [
            _make_result("1", formality="Casual"),
            _make_result("2", formality="Smart Casual"),
            _make_result("3", formality="Formal"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"formality": ["Casual", "Smart Casual"]},
        )
        assert len(filtered) == 2

    def test_null_attribute_excluded(self):
        """Products with None for a filtered attribute are dropped."""
        svc = _service()()
        results = [
            _make_result("1", neckline="V-Neck"),
            _make_result("2", neckline=None),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"neckline": ["V-Neck"]},
        )
        assert len(filtered) == 1


# =============================================================================
# 8. _apply_user_post_filters — multi-value attribute filters
# =============================================================================

class TestUserPostFilterMultiAttrs:
    """Verify multi-value attribute filtering (occasions, materials, etc.)."""

    def test_occasions_filter(self):
        svc = _service()()
        results = [
            _make_result("1", occasions=["everyday", "work"]),
            _make_result("2", occasions=["party", "date night"]),
            _make_result("3", occasions=["work", "casual"]),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"occasions": ["work"]},
        )
        assert len(filtered) == 2
        assert {r["product_id"] for r in filtered} == {"1", "3"}

    def test_materials_filter(self):
        svc = _service()()
        results = [
            _make_result("1", materials=["cotton", "polyester"]),
            _make_result("2", materials=["silk"]),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"materials": ["silk"]},
        )
        assert len(filtered) == 1
        assert filtered[0]["product_id"] == "2"

    def test_seasons_filter(self):
        svc = _service()()
        results = [
            _make_result("1", seasons=["spring", "summer"]),
            _make_result("2", seasons=["fall", "winter"]),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"seasons": ["summer"]},
        )
        assert len(filtered) == 1

    def test_style_tags_filter(self):
        svc = _service()()
        results = [
            _make_result("1", style_tags=["bohemian", "casual"]),
            _make_result("2", style_tags=["minimalist"]),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"style_tags": ["bohemian"]},
        )
        assert len(filtered) == 1

    def test_null_multi_field_excluded(self):
        """Products with None list field are dropped when filter is set."""
        svc = _service()()
        results = [
            _make_result("1", occasions=["work"]),
            _make_result("2", occasions=None),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"occasions": ["work"]},
        )
        assert len(filtered) == 1


# =============================================================================
# 9. _apply_user_post_filters — exclusion filters
# =============================================================================

class TestUserPostFilterExclusions:
    """Verify exclusion filters (exclude_neckline, exclude_colors, etc.)."""

    def test_exclude_neckline(self):
        svc = _service()()
        results = [
            _make_result("1", neckline="V-Neck"),
            _make_result("2", neckline="Strapless"),
            _make_result("3", neckline="Crew"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"exclude_neckline": ["Strapless"]},
        )
        assert len(filtered) == 2
        assert all(r["neckline"] != "Strapless" for r in filtered)

    def test_exclude_sleeve_type(self):
        svc = _service()()
        results = [
            _make_result("1", sleeve_type="Sleeveless"),
            _make_result("2", sleeve_type="Long"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"exclude_sleeve_type": ["Sleeveless"]},
        )
        assert len(filtered) == 1
        assert filtered[0]["sleeve_type"] == "Long"

    def test_exclude_patterns(self):
        svc = _service()()
        results = [
            _make_result("1", pattern="Floral"),
            _make_result("2", pattern="Solid"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"exclude_patterns": ["Floral"]},
        )
        assert len(filtered) == 1
        assert filtered[0]["pattern"] == "Solid"

    def test_exclude_colors_uses_primary_color(self):
        svc = _service()()
        results = [
            _make_result("1", primary_color="Red"),
            _make_result("2", primary_color="Blue"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"exclude_colors": ["Red"]},
        )
        assert len(filtered) == 1
        assert filtered[0]["primary_color"] == "Blue"

    def test_exclude_occasions(self):
        """Exclusion on multi-value field — any match triggers exclusion."""
        svc = _service()()
        results = [
            _make_result("1", occasions=["party", "date night"]),
            _make_result("2", occasions=["work", "everyday"]),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"exclude_occasions": ["party"]},
        )
        assert len(filtered) == 1
        assert filtered[0]["product_id"] == "2"

    def test_null_not_excluded(self):
        """Products with null attribute values survive exclusion filters."""
        svc = _service()()
        results = [
            _make_result("1", neckline=None),
            _make_result("2", neckline="Strapless"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"exclude_neckline": ["Strapless"]},
        )
        # null neckline is not "Strapless" so it survives
        assert len(filtered) == 1
        assert filtered[0]["product_id"] == "1"

    def test_na_not_excluded(self):
        """Products with 'N/A' attribute values survive exclusion filters."""
        svc = _service()()
        results = [
            _make_result("1", neckline="N/A"),
            _make_result("2", neckline="Strapless"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"exclude_neckline": ["Strapless"]},
        )
        assert len(filtered) == 1
        assert filtered[0]["product_id"] == "1"


# =============================================================================
# 10. _apply_user_post_filters — sale and set filters
# =============================================================================

class TestUserPostFilterSaleAndSet:
    """Verify on_sale_only and is_set filtering."""

    def test_on_sale_only(self):
        svc = _service()()
        results = [
            _make_result("1", is_on_sale=True),
            _make_result("2", is_on_sale=False),
            _make_result("3", is_on_sale=True),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"on_sale_only": True},
        )
        assert len(filtered) == 2
        assert all(r["is_on_sale"] for r in filtered)

    def test_is_set_true(self):
        svc = _service()()
        results = [
            _make_result("1", is_set=True),
            _make_result("2", is_set=False),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"is_set": True},
        )
        assert len(filtered) == 1
        assert filtered[0]["is_set"] is True

    def test_is_set_false(self):
        svc = _service()()
        results = [
            _make_result("1", is_set=True),
            _make_result("2", is_set=False),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"is_set": False},
        )
        assert len(filtered) == 1
        assert filtered[0]["product_id"] == "2"


# =============================================================================
# 11. No-op base case — empty user_filters
# =============================================================================

class TestNoOpBaseCase:
    """When user sets no filters, nothing should change."""

    def test_empty_snapshot_returns_empty(self):
        """No user filters → snapshot is empty."""
        svc = _service()
        req = _make_request()
        snap = svc._snapshot_user_filters(req)
        assert snap == {}

    def test_empty_filters_returns_all_results(self):
        """Empty user_filters → all results pass through."""
        svc = _service()()
        results = [
            _make_result("1"),
            _make_result("2"),
            _make_result("3"),
        ]
        filtered = svc._apply_user_post_filters(results, {})
        assert len(filtered) == 3
        assert filtered is results  # same list, no copy

    def test_strip_empty_set_returns_same(self):
        """Stripping nothing returns the same request."""
        svc = _service()
        req = _make_request(brands=["Nike"])
        result = svc._strip_filters_from_request(req, set())
        assert result is req


# =============================================================================
# 12. Combined filter scenario
# =============================================================================

class TestCombinedFilters:
    """Verify multiple user filters applied together."""

    def test_brand_and_price(self):
        """Brand + price range narrows results correctly."""
        svc = _service()()
        results = [
            _make_result("1", brand="Nike", price=30.0),
            _make_result("2", brand="Adidas", price=50.0),
            _make_result("3", brand="Nike", price=80.0),
            _make_result("4", brand="Nike", price=150.0),
        ]
        filtered = svc._apply_user_post_filters(results, {
            "brands": ["Nike"],
            "max_price": 100.0,
        })
        assert len(filtered) == 2
        assert {r["product_id"] for r in filtered} == {"1", "3"}

    def test_brand_and_color_and_formality(self):
        """Three filters intersected."""
        svc = _service()()
        results = [
            _make_result("1", brand="Nike", primary_color="Black", colors=["black"], formality="Casual"),
            _make_result("2", brand="Nike", primary_color="Red", colors=["red"], formality="Casual"),
            _make_result("3", brand="Nike", primary_color="Black", colors=["black"], formality="Formal"),
            _make_result("4", brand="Adidas", primary_color="Black", colors=["black"], formality="Casual"),
        ]
        filtered = svc._apply_user_post_filters(results, {
            "brands": ["Nike"],
            "colors": ["Black"],
            "formality": ["Casual"],
        })
        assert len(filtered) == 1
        assert filtered[0]["product_id"] == "1"

    def test_inclusion_and_exclusion_together(self):
        """Brand inclusion + neckline exclusion."""
        svc = _service()()
        results = [
            _make_result("1", brand="Nike", neckline="V-Neck"),
            _make_result("2", brand="Nike", neckline="Strapless"),
            _make_result("3", brand="Adidas", neckline="V-Neck"),
        ]
        filtered = svc._apply_user_post_filters(results, {
            "brands": ["Nike"],
            "exclude_neckline": ["Strapless"],
        })
        assert len(filtered) == 1
        assert filtered[0]["product_id"] == "1"


# =============================================================================
# 13. The athleisure + Nike scenario — end-to-end logic test
# =============================================================================

class TestAthleisureNikeScenario:
    """
    The bug that triggered this architecture change:

    Query: "athleisure"
    User UI filter: brand=Nike
    Planner guesses: category_l1=["Activewear", "Tops"]

    Old behavior: Algolia filter = brand:"Nike" AND (category_l1:"Activewear" OR
    category_l1:"Tops") → 0 results because Nike items are categorized differently.

    New behavior:
    1. Snapshot captures brands=["Nike"]
    2. Pipeline request has brands=None, category_l1=["Activewear","Tops"]
    3. Algolia searches without Nike brand constraint → gets diverse results
    4. RRF merge produces diverse results
    5. User post-filter keeps only Nike → user gets Nike athleisure items
    """

    def test_snapshot_captures_brand(self):
        svc = _service()
        req = _make_request(query="athleisure", brands=["Nike"])
        snap = svc._snapshot_user_filters(req)
        assert snap == {"brands": ["Nike"]}

    def test_strip_removes_brand_keeps_planner_category(self):
        """After planner adds category_l1, stripping brands keeps category."""
        svc = _service()
        # Simulate: user sets brands, then planner adds category_l1
        req = _make_request(
            query="athleisure",
            brands=["Nike"],
            category_l1=["Activewear", "Tops"],
        )
        pipeline_req = svc._strip_filters_from_request(req, {"brands"})
        assert pipeline_req.brands is None
        assert pipeline_req.category_l1 == ["Activewear", "Tops"]

    def test_post_filter_finds_nike_items(self):
        """Post-filter keeps Nike items regardless of category."""
        svc = _service()()
        # Simulated merged results — diverse brands, some Nike with
        # categories the planner didn't predict
        results = [
            _make_result("1", brand="Nike", category_l1="Sportswear"),
            _make_result("2", brand="Adidas", category_l1="Activewear"),
            _make_result("3", brand="Nike", category_l1="Casual"),
            _make_result("4", brand="Lululemon", category_l1="Activewear"),
            _make_result("5", brand="Nike", category_l1="Activewear"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"brands": ["Nike"]},
        )
        # All 3 Nike items survive, regardless of their category_l1
        assert len(filtered) == 3
        assert all(r["brand"] == "Nike" for r in filtered)

    def test_full_snapshot_strip_postfilter_flow(self):
        """End-to-end: snapshot → strip → post-filter."""
        svc_cls = _service()

        # Step 1: User request
        req = _make_request(query="athleisure", brands=["Nike"])

        # Step 2: Snapshot
        user_filters = svc_cls._snapshot_user_filters(req)
        assert user_filters == {"brands": ["Nike"]}

        # Step 3: Simulate planner adding category_l1
        req_with_planner = req.model_copy(update={
            "category_l1": ["Activewear", "Tops"],
        })

        # Step 4: Strip user filters
        pipeline_req = svc_cls._strip_filters_from_request(
            req_with_planner, set(user_filters.keys()),
        )
        assert pipeline_req.brands is None
        assert pipeline_req.category_l1 == ["Activewear", "Tops"]

        # Step 5: Simulated merged results (diverse brands)
        merged = [
            _make_result("1", brand="Nike", category_l1="Sportswear"),
            _make_result("2", brand="Boohoo", category_l1="Activewear"),
            _make_result("3", brand="Nike", category_l1="Casual"),
            _make_result("4", brand="Nike", category_l1="Activewear"),
        ]

        # Step 6: Post-filter
        svc = svc_cls()
        final = svc._apply_user_post_filters(merged, user_filters)
        assert len(final) == 3
        assert all(r["brand"] == "Nike" for r in final)


# =============================================================================
# 14. Order preservation
# =============================================================================

class TestOrderPreservation:
    """Verify that post-filtering preserves RRF score ordering."""

    def test_order_preserved_after_brand_filter(self):
        svc = _service()()
        results = [
            _make_result("1", brand="Nike", rrf_score=0.9),
            _make_result("2", brand="Adidas", rrf_score=0.8),
            _make_result("3", brand="Nike", rrf_score=0.7),
            _make_result("4", brand="Adidas", rrf_score=0.6),
            _make_result("5", brand="Nike", rrf_score=0.5),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"brands": ["Nike"]},
        )
        scores = [r["rrf_score"] for r in filtered]
        assert scores == [0.9, 0.7, 0.5]

    def test_order_preserved_after_price_filter(self):
        svc = _service()()
        results = [
            _make_result("1", price=100.0, rrf_score=0.9),
            _make_result("2", price=20.0, rrf_score=0.8),
            _make_result("3", price=50.0, rrf_score=0.7),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"max_price": 60.0},
        )
        ids = [r["product_id"] for r in filtered]
        assert ids == ["2", "3"]


# =============================================================================
# 15. Edge cases
# =============================================================================

class TestEdgeCases:
    """Edge cases for the filter architecture."""

    def test_empty_results_list(self):
        """No results → no crash, empty list returned."""
        svc = _service()()
        filtered = svc._apply_user_post_filters(
            [], {"brands": ["Nike"]},
        )
        assert filtered == []

    def test_all_results_filtered_out(self):
        """All results fail the filter → empty list."""
        svc = _service()()
        results = [
            _make_result("1", brand="Adidas"),
            _make_result("2", brand="Puma"),
        ]
        filtered = svc._apply_user_post_filters(
            results, {"brands": ["Nike"]},
        )
        assert filtered == []

    def test_no_matching_field_in_result_dict(self):
        """Result dict missing the field entirely → excluded."""
        svc = _service()()
        results = [
            {"product_id": "1", "price": 30.0, "rrf_score": 0.5},
        ]
        filtered = svc._apply_user_post_filters(
            results, {"brands": ["Nike"]},
        )
        assert filtered == []

    def test_price_as_string_coerced(self):
        """Price stored as string should still work."""
        svc = _service()()
        results = [
            {**_make_result("1"), "price": "50.0"},
            {**_make_result("2"), "price": "150.0"},
        ]
        filtered = svc._apply_user_post_filters(
            results, {"max_price": 100.0},
        )
        assert len(filtered) == 1
        assert filtered[0]["product_id"] == "1"


# =============================================================================
# 16. _compute_facets_from_results
# =============================================================================

class TestComputeFacetsFromResults:
    """Verify facet computation from merged result sets.

    Mirrors Algolia's facet processing:
    - count > 1 to include a value
    - null/N/A/none/"" excluded
    - field needs >= 2 distinct valid values
    - sorted by count descending
    """

    def test_empty_results_returns_none(self):
        """No results → None facets."""
        svc = _service()()
        result = svc._compute_facets_from_results([])
        assert result is None

    def test_brand_facet_computed(self):
        """Brands counted from scalar 'brand' field."""
        svc = _service()()
        results = [
            _make_result("1", brand="Nike"),
            _make_result("2", brand="Nike"),
            _make_result("3", brand="Adidas"),
            _make_result("4", brand="Adidas"),
            _make_result("5", brand="Puma"),
            _make_result("6", brand="Puma"),
        ]
        facets = svc._compute_facets_from_results(results)
        assert facets is not None
        assert "brand" in facets
        brand_vals = {fv.value: fv.count for fv in facets["brand"]}
        assert brand_vals == {"Nike": 2, "Adidas": 2, "Puma": 2}

    def test_facet_sorted_by_count_descending(self):
        """Values must be sorted highest count first."""
        svc = _service()()
        results = [
            _make_result("1", brand="Nike"),
            _make_result("2", brand="Nike"),
            _make_result("3", brand="Nike"),
            _make_result("4", brand="Adidas"),
            _make_result("5", brand="Adidas"),
            _make_result("6", brand="Puma"),
            _make_result("7", brand="Puma"),
        ]
        facets = svc._compute_facets_from_results(results)
        brand_facet = facets["brand"]
        counts = [fv.count for fv in brand_facet]
        assert counts == sorted(counts, reverse=True)
        assert brand_facet[0].value == "Nike"
        assert brand_facet[0].count == 3

    def test_count_one_excluded(self):
        """Values appearing only once are excluded (count > 1 rule)."""
        svc = _service()()
        results = [
            _make_result("1", brand="Nike"),
            _make_result("2", brand="Nike"),
            _make_result("3", brand="Adidas"),  # count=1 → excluded
            _make_result("4", brand="Puma"),
            _make_result("5", brand="Puma"),
        ]
        facets = svc._compute_facets_from_results(results)
        brand_vals = {fv.value for fv in facets["brand"]}
        assert "Nike" in brand_vals
        assert "Puma" in brand_vals
        assert "Adidas" not in brand_vals

    def test_null_values_excluded(self):
        """Null-like values filtered out."""
        svc = _service()()
        results = [
            _make_result("1", pattern="Solid"),
            _make_result("2", pattern="Solid"),
            _make_result("3", pattern="Floral"),
            _make_result("4", pattern="Floral"),
            _make_result("5", pattern="N/A"),
            _make_result("6", pattern="N/A"),
            _make_result("7", pattern=None),
            _make_result("8", pattern="null"),
        ]
        facets = svc._compute_facets_from_results(results)
        pattern_vals = {fv.value for fv in facets["pattern"]}
        assert "Solid" in pattern_vals
        assert "Floral" in pattern_vals
        assert "N/A" not in pattern_vals
        assert "null" not in pattern_vals

    def test_field_with_one_distinct_value_excluded(self):
        """A facet field with only 1 valid value is not included (>= 2 rule)."""
        svc = _service()()
        # All items have same formality → field excluded
        results = [
            _make_result("1", formality="Casual", brand="Nike"),
            _make_result("2", formality="Casual", brand="Nike"),
            _make_result("3", formality="Casual", brand="Adidas"),
            _make_result("4", formality="Casual", brand="Adidas"),
        ]
        facets = svc._compute_facets_from_results(results)
        # formality has only "Casual" (count=4) → only 1 distinct → excluded
        assert "formality" not in facets

    def test_multi_value_field_occasions(self):
        """Multi-value fields: each list element counted individually."""
        svc = _service()()
        results = [
            _make_result("1", occasions=["work", "everyday"]),
            _make_result("2", occasions=["work", "date night"]),
            _make_result("3", occasions=["everyday", "casual"]),
            _make_result("4", occasions=["work", "everyday"]),
        ]
        facets = svc._compute_facets_from_results(results)
        assert "occasions" in facets
        occ_vals = {fv.value: fv.count for fv in facets["occasions"]}
        assert occ_vals["work"] == 3
        assert occ_vals["everyday"] == 3
        # "date night" count=1, "casual" count=1 → excluded
        assert "date night" not in occ_vals
        assert "casual" not in occ_vals

    def test_multi_value_field_materials(self):
        """Materials list counted per element."""
        svc = _service()()
        results = [
            _make_result("1", materials=["cotton", "polyester"]),
            _make_result("2", materials=["cotton", "silk"]),
            _make_result("3", materials=["cotton"]),
            _make_result("4", materials=["silk", "polyester"]),
        ]
        facets = svc._compute_facets_from_results(results)
        assert "materials" in facets
        mat_vals = {fv.value: fv.count for fv in facets["materials"]}
        assert mat_vals["cotton"] == 3
        assert mat_vals["polyester"] == 2
        assert mat_vals["silk"] == 2

    def test_multi_value_field_seasons(self):
        """Seasons list counted per element."""
        svc = _service()()
        results = [
            _make_result("1", seasons=["spring", "summer"]),
            _make_result("2", seasons=["spring", "fall"]),
            _make_result("3", seasons=["summer", "spring"]),
        ]
        facets = svc._compute_facets_from_results(results)
        assert "seasons" in facets
        seas_vals = {fv.value: fv.count for fv in facets["seasons"]}
        assert seas_vals["spring"] == 3
        assert seas_vals["summer"] == 2
        # "fall" count=1 → excluded
        assert "fall" not in seas_vals

    def test_multi_value_field_style_tags(self):
        """Style tags list counted per element."""
        svc = _service()()
        results = [
            _make_result("1", style_tags=["bohemian", "casual"]),
            _make_result("2", style_tags=["bohemian", "minimalist"]),
            _make_result("3", style_tags=["casual", "bohemian"]),
        ]
        facets = svc._compute_facets_from_results(results)
        assert "style_tags" in facets
        st_vals = {fv.value: fv.count for fv in facets["style_tags"]}
        assert st_vals["bohemian"] == 3
        assert st_vals["casual"] == 2
        # "minimalist" count=1 → excluded
        assert "minimalist" not in st_vals

    def test_is_on_sale_boolean_to_string(self):
        """is_on_sale boolean → 'true'/'false' string facet values."""
        svc = _service()()
        results = [
            _make_result("1", is_on_sale=True),
            _make_result("2", is_on_sale=True),
            _make_result("3", is_on_sale=False),
            _make_result("4", is_on_sale=False),
        ]
        facets = svc._compute_facets_from_results(results)
        assert "is_on_sale" in facets
        sale_vals = {fv.value: fv.count for fv in facets["is_on_sale"]}
        assert sale_vals["true"] == 2
        assert sale_vals["false"] == 2

    def test_category_l1_facet(self):
        svc = _service()()
        results = [
            _make_result("1", category_l1="Tops"),
            _make_result("2", category_l1="Tops"),
            _make_result("3", category_l1="Dresses"),
            _make_result("4", category_l1="Dresses"),
            _make_result("5", category_l1="Bottoms"),
            _make_result("6", category_l1="Bottoms"),
        ]
        facets = svc._compute_facets_from_results(results)
        assert "category_l1" in facets
        cat_vals = {fv.value: fv.count for fv in facets["category_l1"]}
        assert cat_vals == {"Tops": 2, "Dresses": 2, "Bottoms": 2}

    def test_primary_color_facet(self):
        svc = _service()()
        results = [
            _make_result("1", primary_color="Black"),
            _make_result("2", primary_color="Black"),
            _make_result("3", primary_color="White"),
            _make_result("4", primary_color="White"),
            _make_result("5", primary_color="Red"),
            _make_result("6", primary_color="Red"),
        ]
        facets = svc._compute_facets_from_results(results)
        assert "primary_color" in facets
        assert len(facets["primary_color"]) == 3

    def test_all_19_facet_fields_present_when_diverse(self):
        """When results have diverse values for all fields, all 19 appear."""
        svc = _service()()
        # Build two results that differ in every facet field
        r1 = _make_result(
            "1",
            brand="Nike",
            category_l1="Tops",
            broad_category="tops",
            article_type="t-shirt",
            formality="Casual",
            primary_color="Black",
            color_family="Dark",
            pattern="Solid",
            fit_type="Regular",
            neckline="Crew",
            sleeve_type="Short",
            length="Regular",
            silhouette="Straight",
            rise="Mid",
            occasions=["everyday"],
            seasons=["spring"],
            style_tags=["casual"],
            materials=["cotton"],
            is_on_sale=True,
        )
        r2 = _make_result(
            "2",
            brand="Nike",
            category_l1="Tops",
            broad_category="tops",
            article_type="t-shirt",
            formality="Casual",
            primary_color="Black",
            color_family="Dark",
            pattern="Solid",
            fit_type="Regular",
            neckline="Crew",
            sleeve_type="Short",
            length="Regular",
            silhouette="Straight",
            rise="Mid",
            occasions=["everyday"],
            seasons=["spring"],
            style_tags=["casual"],
            materials=["cotton"],
            is_on_sale=True,
        )
        r3 = _make_result(
            "3",
            brand="Adidas",
            category_l1="Dresses",
            broad_category="dresses",
            article_type="dress",
            formality="Formal",
            primary_color="Red",
            color_family="Warm",
            pattern="Floral",
            fit_type="Fitted",
            neckline="V-Neck",
            sleeve_type="Long",
            length="Midi",
            silhouette="A-Line",
            rise="High",
            occasions=["party"],
            seasons=["winter"],
            style_tags=["elegant"],
            materials=["silk"],
            is_on_sale=False,
        )
        r4 = _make_result(
            "4",
            brand="Adidas",
            category_l1="Dresses",
            broad_category="dresses",
            article_type="dress",
            formality="Formal",
            primary_color="Red",
            color_family="Warm",
            pattern="Floral",
            fit_type="Fitted",
            neckline="V-Neck",
            sleeve_type="Long",
            length="Midi",
            silhouette="A-Line",
            rise="High",
            occasions=["party"],
            seasons=["winter"],
            style_tags=["elegant"],
            materials=["silk"],
            is_on_sale=False,
        )
        facets = svc._compute_facets_from_results([r1, r2, r3, r4])
        assert facets is not None
        expected_keys = {
            "brand", "category_l1", "broad_category", "article_type",
            "formality", "primary_color", "color_family", "pattern",
            "fit_type", "neckline", "sleeve_type", "length", "silhouette",
            "rise", "occasions", "seasons", "style_tags", "materials",
            "is_on_sale",
        }
        assert set(facets.keys()) == expected_keys

    def test_missing_field_in_result_dict_skipped(self):
        """Result dicts missing a field don't crash — just skip that entry."""
        svc = _service()()
        results = [
            _make_result("1", brand="Nike"),
            _make_result("2", brand="Nike"),
            {"product_id": "3", "brand": "Adidas", "price": 30.0},
            {"product_id": "4", "brand": "Adidas", "price": 40.0},
        ]
        facets = svc._compute_facets_from_results(results)
        assert "brand" in facets
        brand_vals = {fv.value: fv.count for fv in facets["brand"]}
        assert brand_vals == {"Nike": 2, "Adidas": 2}

    def test_facet_values_preserve_original_case(self):
        """Facet values should keep original casing (not lowercased)."""
        svc = _service()()
        results = [
            _make_result("1", brand="Princess Polly"),
            _make_result("2", brand="Princess Polly"),
            _make_result("3", brand="ASOS"),
            _make_result("4", brand="ASOS"),
        ]
        facets = svc._compute_facets_from_results(results)
        brand_vals = {fv.value for fv in facets["brand"]}
        assert "Princess Polly" in brand_vals
        assert "ASOS" in brand_vals
