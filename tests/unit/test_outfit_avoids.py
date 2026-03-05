"""Unit tests for the outfit avoids system (hard filters + soft penalties).

Uses SimpleNamespace mocks to avoid importing the real AestheticProfile
dataclass (which would pull in heavy ML dependencies).
"""

import pytest
from types import SimpleNamespace
from typing import Optional, Set

from services.outfit_avoids import (
    PENALTY_CAP,
    compute_avoid_penalties,
    filter_hard_avoids,
    # Type-bucket classifiers (tested indirectly through rules)
    _is_formal_evening,
    _is_sporty_casual,
    _is_party_fabric,
    _is_tailored,
    _is_gym_piece,
    _is_activewear,
    _is_formal_piece,
    # Lifestyle context classifiers
    _has_active_context,
    _has_polished_context,
    # Individual rule functions
    _check_A1, _check_A2, _check_A3, _check_A4,
    _check_B1,
    _check_C1, _check_C2,
    _check_D1,
    _check_E1, _check_E2, _check_E3,
    _check_F1, _check_F2, _check_F3,
    _check_G1, _check_G2, _check_G3,
    _check_I1, _check_I2, _check_I3, _check_I4,
    _check_J1, _check_J2,
    _check_K1,
    _check_L1, _check_L2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _p(**kwargs) -> SimpleNamespace:
    """Create a mock AestheticProfile with sensible defaults."""
    defaults = {
        "product_id": "test-001",
        "name": "Test Product",
        "brand": "TestBrand",
        "category": "tops",
        "broad_category": "tops",
        "price": 50.0,
        "image_url": None,
        "gemini_category_l1": "Tops",
        "gemini_category_l2": "t-shirt",
        "formality": "casual",
        "formality_level": 2,
        "occasions": [],
        "style_tags": [],
        "pattern": "solid",
        "fit_type": "regular",
        "color_family": "black",
        "primary_color": "black",
        "secondary_colors": [],
        "seasons": [],
        "silhouette": "regular",
        "apparent_fabric": "cotton",
        "texture": "smooth",
        "coverage_level": "moderate",
        "sheen": "matte",
        "rise": None,
        "leg_shape": None,
        "stretch": None,
        "length": None,
        # Derived
        "is_bridge": False,
        "primary_style": None,
        "style_strength": 0.35,
        "material_family": "cotton",
        "texture_intensity": "smooth",
        "shine_level": "matte",
        "fabric_weight": "mid",
        "layer_role": "base",
        "temp_band": "mild",
        "color_saturation": "medium",
        "similarity": 0.5,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# Shorthand factories for common item archetypes
def _hoodie(**kw):
    return _p(
        gemini_category_l2="hoodie", formality_level=1,
        material_family="knit", fabric_weight="heavy",
        layer_role="midlayer", style_tags=["casual"],
        **kw,
    )

def _cocktail_dress(**kw):
    return _p(
        gemini_category_l2="cocktail dress", formality_level=5,
        material_family="satin", fabric_weight="light",
        broad_category="dresses", layer_role="base",
        style_tags=["glamorous"], sheen="satin", shine_level="slight",
        **kw,
    )

def _blazer(**kw):
    return _p(
        gemini_category_l2="blazer", formality_level=3,
        material_family="wool", fabric_weight="mid",
        broad_category="outerwear", layer_role="outer",
        style_tags=["classic"], **kw,
    )

def _track_pants(**kw):
    return _p(
        gemini_category_l2="track pants", formality_level=1,
        material_family="technical", fabric_weight="light",
        broad_category="bottoms", style_tags=["sporty"],
        **kw,
    )

def _performance_legging(**kw):
    return _p(
        gemini_category_l2="performance legging", formality_level=1,
        material_family="synthetic_stretch", fabric_weight="light",
        broad_category="bottoms", style_tags=["athletic"],
        **kw,
    )

def _sequin_top(**kw):
    return _p(
        gemini_category_l2="sequin top", formality_level=4,
        material_family="sequin", fabric_weight="mid",
        shine_level="shiny", sheen="metallic",
        style_tags=["party", "glamorous"], **kw,
    )

def _denim_jeans(**kw):
    return _p(
        gemini_category_l2="jeans", formality_level=2,
        material_family="denim", fabric_weight="mid",
        broad_category="bottoms", pattern="solid", **kw,
    )

def _casual_top(**kw):
    return _p(
        gemini_category_l2="t-shirt", formality_level=2,
        material_family="cotton", fabric_weight="mid",
        broad_category="tops", layer_role="base",
        silhouette="regular", fit_type="regular",
        style_tags=["casual"], **kw,
    )

def _wide_leg_pants(**kw):
    return _p(
        gemini_category_l2="wide-leg trousers", formality_level=2,
        material_family="cotton", fabric_weight="mid",
        broad_category="bottoms", silhouette="wide leg",
        **kw,
    )


# =========================================================================
# TYPE-BUCKET CLASSIFIER TESTS
# =========================================================================

class TestTypeBucketClassifiers:

    def test_is_formal_evening_by_formality(self):
        assert _is_formal_evening(_p(formality_level=4)) is True
        assert _is_formal_evening(_p(formality_level=5)) is True
        assert _is_formal_evening(_p(formality_level=3)) is False

    def test_is_formal_evening_by_l2(self):
        assert _is_formal_evening(_p(formality_level=2, gemini_category_l2="cocktail dress")) is True
        assert _is_formal_evening(_p(formality_level=2, gemini_category_l2="gown")) is True

    def test_is_sporty_casual_by_l2(self):
        assert _is_sporty_casual(_hoodie()) is True
        assert _is_sporty_casual(_p(gemini_category_l2="sweatshirt")) is True
        assert _is_sporty_casual(_p(gemini_category_l2="jogger")) is True

    def test_is_sporty_casual_by_technical_material(self):
        assert _is_sporty_casual(_p(material_family="technical", style_tags=["sporty"])) is True
        assert _is_sporty_casual(_p(material_family="technical", style_tags=["classic"])) is False

    def test_is_party_fabric_by_material(self):
        assert _is_party_fabric(_sequin_top()) is True

    def test_is_party_fabric_by_sheen(self):
        assert _is_party_fabric(_p(sheen="metallic", formality_level=3)) is True
        assert _is_party_fabric(_p(sheen="metallic", formality_level=1)) is False

    def test_is_tailored(self):
        assert _is_tailored(_blazer()) is True
        assert _is_tailored(_p(formality_level=3, gemini_category_l2="pencil skirt")) is True
        assert _is_tailored(_hoodie()) is False

    def test_is_gym_piece(self):
        assert _is_gym_piece(_track_pants()) is True
        assert _is_gym_piece(_p(gemini_category_l2="windbreaker")) is True

    def test_is_activewear(self):
        assert _is_activewear(_performance_legging()) is True
        assert _is_activewear(_casual_top()) is False

    def test_is_formal_piece(self):
        assert _is_formal_piece(_p(formality_level=4, material_family="silk")) is True
        assert _is_formal_piece(_p(formality_level=4, material_family="cotton")) is False
        assert _is_formal_piece(_p(formality_level=2, material_family="silk")) is False


# =========================================================================
# HARD-FILTER TESTS
# =========================================================================

class TestHardFilters:

    def test_hf1_formal_evening_plus_sporty_casual(self):
        """Cocktail dress + hoodie → filtered out."""
        source = _cocktail_dress()
        candidates = [_hoodie(), _casual_top()]
        result = filter_hard_avoids(source, candidates)
        assert len(result) == 1
        assert result[0].gemini_category_l2 == "t-shirt"

    def test_hf1_reverse_sporty_source_formal_candidate(self):
        """Hoodie source + formal evening candidate → filtered."""
        source = _hoodie()
        candidates = [_cocktail_dress(), _denim_jeans()]
        result = filter_hard_avoids(source, candidates)
        assert len(result) == 1
        assert result[0].gemini_category_l2 == "jeans"

    def test_hf1_not_triggered_moderate_formality(self):
        """Formality 3 is NOT formal evening — no filter."""
        source = _p(formality_level=3, gemini_category_l2="midi dress")
        candidates = [_hoodie()]
        result = filter_hard_avoids(source, candidates)
        assert len(result) == 1

    def test_hf1_overridden_by_eclectic_style(self):
        """Eclectic style overrides HF1."""
        source = _cocktail_dress()
        candidates = [_hoodie()]
        result = filter_hard_avoids(source, candidates, user_styles={"eclectic"})
        assert len(result) == 1

    def test_hf2_sporty_casual_plus_party_fabric(self):
        """Hoodie + sequin top → filtered."""
        source = _hoodie()
        candidates = [_sequin_top(), _casual_top()]
        result = filter_hard_avoids(source, candidates)
        assert len(result) == 1
        assert result[0].gemini_category_l2 == "t-shirt"

    def test_hf2_overridden_by_edgy_style(self):
        """Edgy style overrides HF2."""
        source = _hoodie()
        # Party-fabric with formality 2 so HF1 (formal↔sporty) does NOT fire
        cand = _p(
            material_family="sequin", sheen="metallic", formality_level=2,
            gemini_category_l2="sequin skirt",
        )
        result = filter_hard_avoids(source, [cand], user_styles={"edgy"})
        assert len(result) == 1

    def test_hf3_tailored_plus_gym(self):
        """Blazer + track pants → filtered."""
        source = _blazer()
        candidates = [_track_pants(), _denim_jeans()]
        result = filter_hard_avoids(source, candidates)
        assert len(result) == 1
        assert result[0].gemini_category_l2 == "jeans"

    def test_hf3_overridden_by_athleisure(self):
        """Athleisure style overrides HF3."""
        source = _blazer()
        candidates = [_track_pants()]
        result = filter_hard_avoids(source, candidates, user_styles={"athleisure"})
        assert len(result) == 1

    def test_hf4_activewear_plus_formal_no_override(self):
        """Performance legging + formal silk top → always filtered."""
        formal_top = _p(
            formality_level=4, material_family="silk",
            broad_category="tops", gemini_category_l2="formal blouse",
        )
        source = _performance_legging()
        candidates = [formal_top, _casual_top()]
        # No style override can disable HF4
        result = filter_hard_avoids(source, candidates, user_styles={"edgy", "eclectic"})
        assert len(result) == 1
        assert result[0].gemini_category_l2 == "t-shirt"

    def test_hard_filter_preserves_good_candidates(self):
        """Compatible candidates pass through untouched."""
        source = _casual_top()
        candidates = [_denim_jeans(), _wide_leg_pants(), _blazer()]
        result = filter_hard_avoids(source, candidates)
        assert len(result) == 3


# =========================================================================
# SOFT-PENALTY RULE TESTS
# =========================================================================

class TestRuleA_Layering:

    def test_A1_light_layer_over_heavy_base(self):
        """Thin cardigan (light, midlayer) over hoodie (heavy) → -0.10."""
        src = _hoodie()  # already fabric_weight="heavy"
        cand = _p(
            fabric_weight="light", layer_role="midlayer",
            gemini_category_l2="cardigan",
        )
        assert _check_A1(src, cand) == pytest.approx(-0.10)

    def test_A1_no_penalty_heavy_over_heavy(self):
        src = _hoodie()  # already fabric_weight="heavy"
        cand = _p(fabric_weight="heavy", layer_role="outer")
        assert _check_A1(src, cand) == 0.0

    def test_A1_no_penalty_mid_over_light(self):
        src = _p(fabric_weight="light", layer_role="base")
        cand = _p(fabric_weight="mid", layer_role="midlayer")
        assert _check_A1(src, cand) == 0.0

    def test_A2_cold_outer_hot_anchor(self):
        """Puffer coat (cold, outer) over summer tank (hot) → -0.12."""
        src = _p(temp_band="hot", layer_role="base")
        cand = _p(temp_band="cold", layer_role="outer")
        assert _check_A2(src, cand) == pytest.approx(-0.12)

    def test_A2_no_penalty_same_band(self):
        src = _p(temp_band="mild", layer_role="base")
        cand = _p(temp_band="mild", layer_role="outer")
        assert _check_A2(src, cand) == 0.0

    def test_A3_hot_layer_cold_anchor(self):
        """Kimono (hot, midlayer) over chunky sweater (cold) → -0.08."""
        src = _p(temp_band="cold", layer_role="base")
        cand = _p(temp_band="hot", layer_role="midlayer")
        assert _check_A3(src, cand) == pytest.approx(-0.08)

    def test_A3_no_penalty_base_roles(self):
        src = _p(temp_band="cold", layer_role="base")
        cand = _p(temp_band="hot", layer_role="base")
        assert _check_A3(src, cand) == 0.0


class TestRuleA4_FlimsyOuterwear:
    """A4: Flimsy outerwear over lightweight top — no layering purpose."""

    def test_knit_cardigan_over_tank_top(self):
        """Knit cardigan over a partial-coverage tank → -0.12."""
        src = _p(
            gemini_category_l1="Tops", gemini_category_l2="Tank Top",
            coverage_level="Partial", fabric_weight="light",
            apparent_fabric="Cotton",
        )
        cand = _p(
            gemini_category_l1="Outerwear", gemini_category_l2="Cardigan",
            apparent_fabric="Knit", fabric_weight="light",
        )
        assert _check_A4(src, cand) == pytest.approx(-0.12)

    def test_knit_cardigan_over_partial_top(self):
        """Knit cardigan over a partial-coverage top with mid weight → -0.12."""
        src = _p(
            gemini_category_l1="Tops", gemini_category_l2="Top",
            coverage_level="Partial", fabric_weight="mid",
            apparent_fabric="Cotton",
        )
        cand = _p(
            gemini_category_l1="Outerwear", gemini_category_l2="Cardigan",
            apparent_fabric="Knit", fabric_weight="",
        )
        assert _check_A4(src, cand) == pytest.approx(-0.12)

    def test_crochet_shrug_over_camisole(self):
        """Crochet shrug over camisole → -0.12."""
        src = _p(
            gemini_category_l1="Tops", gemini_category_l2="Camisole",
            coverage_level="Partial", fabric_weight="light",
        )
        cand = _p(
            gemini_category_l1="Outerwear", gemini_category_l2="Shrug",
            apparent_fabric="Crochet", fabric_weight="light",
        )
        assert _check_A4(src, cand) == pytest.approx(-0.12)

    def test_blazer_over_tank_no_penalty(self):
        """Structured blazer over tank top → no penalty (not flimsy)."""
        src = _p(
            gemini_category_l1="Tops", gemini_category_l2="Tank Top",
            coverage_level="Partial", fabric_weight="light",
        )
        cand = _p(
            gemini_category_l1="Outerwear", gemini_category_l2="Blazer",
            apparent_fabric="Cotton", fabric_weight="mid",
        )
        assert _check_A4(src, cand) == 0.0

    def test_denim_jacket_over_top_no_penalty(self):
        """Denim jacket over lightweight top → no penalty."""
        src = _p(
            gemini_category_l1="Tops", gemini_category_l2="Top",
            coverage_level="Partial", fabric_weight="light",
        )
        cand = _p(
            gemini_category_l1="Outerwear", gemini_category_l2="Denim Jacket",
            apparent_fabric="Denim", fabric_weight="mid",
        )
        assert _check_A4(src, cand) == 0.0

    def test_heavy_wool_cardigan_no_penalty(self):
        """Heavy wool cardigan → no penalty (provides warmth)."""
        src = _p(
            gemini_category_l1="Tops", gemini_category_l2="Top",
            coverage_level="Partial", fabric_weight="light",
        )
        cand = _p(
            gemini_category_l1="Outerwear", gemini_category_l2="Cardigan",
            apparent_fabric="Wool", fabric_weight="heavy",
        )
        assert _check_A4(src, cand) == 0.0

    def test_cardigan_over_sweater_no_penalty(self):
        """Cardigan over a sweater (not lightweight top) → no penalty."""
        src = _p(
            gemini_category_l1="Tops", gemini_category_l2="Sweater",
            coverage_level="Full", fabric_weight="mid",
        )
        cand = _p(
            gemini_category_l1="Outerwear", gemini_category_l2="Cardigan",
            apparent_fabric="Knit", fabric_weight="light",
        )
        assert _check_A4(src, cand) == 0.0

    def test_symmetric_direction(self):
        """Rule fires in both directions."""
        top = _p(
            gemini_category_l1="Tops", gemini_category_l2="Crop Top",
            coverage_level="Partial", fabric_weight="light",
        )
        cardi = _p(
            gemini_category_l1="Outerwear", gemini_category_l2="Cardigan",
            apparent_fabric="Knit", fabric_weight="light",
        )
        assert _check_A4(top, cardi) == pytest.approx(-0.12)
        assert _check_A4(cardi, top) == pytest.approx(-0.12)

    def test_both_outerwear_no_penalty(self):
        """Two outerwear items → no penalty (neither is a lightweight top)."""
        jacket = _p(
            gemini_category_l1="Outerwear", gemini_category_l2="Blazer",
            apparent_fabric="Cotton", fabric_weight="mid",
        )
        cardi = _p(
            gemini_category_l1="Outerwear", gemini_category_l2="Cardigan",
            apparent_fabric="Knit", fabric_weight="light",
        )
        assert _check_A4(jacket, cardi) == 0.0

    def test_bottoms_not_affected(self):
        """Bottoms source → no penalty (not a lightweight top)."""
        src = _p(
            gemini_category_l1="Bottoms", gemini_category_l2="Shorts",
            coverage_level="Partial", fabric_weight="light",
        )
        cand = _p(
            gemini_category_l1="Outerwear", gemini_category_l2="Cardigan",
            apparent_fabric="Knit", fabric_weight="light",
        )
        assert _check_A4(src, cand) == 0.0

    def test_null_attributes_no_crash(self):
        """Missing attributes don't crash."""
        src = _p(gemini_category_l1=None, gemini_category_l2=None, coverage_level=None, fabric_weight=None)
        cand = _p(gemini_category_l1=None, gemini_category_l2=None, apparent_fabric=None, fabric_weight=None)
        assert _check_A4(src, cand) == 0.0

    def test_kimono_over_tube_top(self):
        """Lightweight kimono over tube top → -0.12."""
        src = _p(
            gemini_category_l1="Tops", gemini_category_l2="Tube Top",
            coverage_level="Partial", fabric_weight="light",
        )
        cand = _p(
            gemini_category_l1="Outerwear", gemini_category_l2="Kimono",
            apparent_fabric="Chiffon", fabric_weight="light",
        )
        assert _check_A4(src, cand) == pytest.approx(-0.12)


class TestRuleB_Formality:

    def test_B1_formality_gap_3(self):
        """Delta 3 → -0.08."""
        src = _p(formality_level=1)
        cand = _p(formality_level=4)
        assert _check_B1(src, cand) == pytest.approx(-0.08)

    def test_B1_formality_gap_4(self):
        """Delta 4 → -0.15."""
        src = _p(formality_level=1)
        cand = _p(formality_level=5)
        assert _check_B1(src, cand) == pytest.approx(-0.15)

    def test_B1_formality_gap_2_no_penalty(self):
        src = _p(formality_level=1)
        cand = _p(formality_level=3)
        assert _check_B1(src, cand) == 0.0

    def test_B1_same_formality(self):
        src = _p(formality_level=3)
        cand = _p(formality_level=3)
        assert _check_B1(src, cand) == 0.0


class TestRuleC_SportyTailored:

    def test_C1_technical_plus_tailored(self):
        """Technical sporty material + wool formal → -0.10."""
        src = _p(material_family="technical", style_tags=["sporty"], formality_level=1)
        cand = _p(material_family="wool", formality_level=3)
        assert _check_C1(src, cand) == pytest.approx(-0.10)

    def test_C1_no_penalty_without_sporty_tag(self):
        src = _p(material_family="technical", style_tags=["casual"], formality_level=1)
        cand = _p(material_family="wool", formality_level=3)
        assert _check_C1(src, cand) == 0.0

    def test_C2_sporty_l2_plus_formal_l2(self):
        """Track top + blazer → -0.10."""
        src = _p(gemini_category_l2="track top")
        cand = _p(gemini_category_l2="blazer")
        assert _check_C2(src, cand) == pytest.approx(-0.10)

    def test_C2_reverse(self):
        src = _p(gemini_category_l2="pencil skirt")
        cand = _p(gemini_category_l2="jogger")
        assert _check_C2(src, cand) == pytest.approx(-0.10)

    def test_C2_no_penalty_both_casual(self):
        src = _p(gemini_category_l2="t-shirt")
        cand = _p(gemini_category_l2="jeans")
        assert _check_C2(src, cand) == 0.0


class TestRuleD_SameMaterial:

    def test_D1_denim_on_denim(self):
        src = _denim_jeans()
        cand = _p(material_family="denim", gemini_category_l2="denim jacket")
        assert _check_D1(src, cand) == pytest.approx(-0.10)

    def test_D1_leather_on_leather(self):
        src = _p(material_family="leather")
        cand = _p(material_family="leather")
        assert _check_D1(src, cand) == pytest.approx(-0.10)

    def test_D1_sequin_on_sequin(self):
        src = _p(material_family="sequin")
        cand = _p(material_family="sequin")
        assert _check_D1(src, cand) == pytest.approx(-0.10)

    def test_D1_tweed_on_tweed(self):
        """Wool + tweed texture both sides → penalty."""
        src = _p(material_family="wool", texture="tweed")
        cand = _p(material_family="wool", texture="boucle")
        assert _check_D1(src, cand) == pytest.approx(-0.10)

    def test_D1_wool_plain_no_penalty(self):
        """Plain wool + plain wool → no penalty (not statement texture)."""
        src = _p(material_family="wool", texture="smooth")
        cand = _p(material_family="wool", texture="smooth")
        assert _check_D1(src, cand) == 0.0

    def test_D1_different_materials_no_penalty(self):
        src = _p(material_family="denim")
        cand = _p(material_family="cotton")
        assert _check_D1(src, cand) == 0.0

    def test_D1_missing_material_no_penalty(self):
        src = _p(material_family=None)
        cand = _p(material_family=None)
        assert _check_D1(src, cand) == 0.0


class TestRuleE_Silhouette:

    def test_E1_both_oversized_not_outerwear(self):
        src = _p(silhouette="oversized", broad_category="tops")
        cand = _p(silhouette="relaxed", broad_category="bottoms")
        assert _check_E1(src, cand) == pytest.approx(-0.08)

    def test_E1_fit_type_fallback(self):
        """Falls back to fit_type when silhouette is generic."""
        src = _p(silhouette="regular", fit_type="oversized", broad_category="tops")
        cand = _p(silhouette="regular", fit_type="baggy", broad_category="bottoms")
        assert _check_E1(src, cand) == pytest.approx(-0.08)

    def test_E1_outerwear_exemption(self):
        """Outerwear gets a pass on oversized."""
        src = _p(silhouette="oversized", broad_category="outerwear")
        cand = _p(silhouette="oversized", broad_category="tops")
        assert _check_E1(src, cand) == 0.0

    def test_E1_one_fitted_no_penalty(self):
        src = _p(silhouette="oversized", broad_category="tops")
        cand = _p(silhouette="fitted", broad_category="bottoms")
        assert _check_E1(src, cand) == 0.0

    def test_E2_cropped_outer_over_longline_heavy(self):
        src = _p(fabric_weight="heavy", length="long", layer_role="base")
        cand = _p(layer_role="outer", length="cropped")
        assert _check_E2(src, cand) == pytest.approx(-0.08)

    def test_E2_no_penalty_long_outer(self):
        src = _p(fabric_weight="heavy", length="long", layer_role="base")
        cand = _p(layer_role="outer", length="long")
        assert _check_E2(src, cand) == 0.0

    def test_E3_bulky_top_wide_bottom(self):
        src = _p(
            silhouette="oversized", fabric_weight="heavy", broad_category="tops",
        )
        cand = _p(silhouette="wide leg", broad_category="bottoms")
        assert _check_E3(src, cand) == pytest.approx(-0.06)

    def test_E3_no_penalty_light_top(self):
        """Light oversized top + wide bottom → no E3 (fabric not heavy)."""
        src = _p(
            silhouette="oversized", fabric_weight="light", broad_category="tops",
        )
        cand = _p(silhouette="wide leg", broad_category="bottoms")
        assert _check_E3(src, cand) == 0.0


class TestRuleF_TextureMaterialClash:

    def test_F1_party_fabric_plus_casual(self):
        """Sequin top + hoodie → -0.12."""
        src = _sequin_top()
        cand = _hoodie()
        assert _check_F1(src, cand) == pytest.approx(-0.12)

    def test_F1_shiny_plus_jogger(self):
        src = _p(shine_level="shiny", sheen="metallic")
        cand = _p(gemini_category_l2="jogger")
        assert _check_F1(src, cand) == pytest.approx(-0.12)

    def test_F1_no_penalty_matte_plus_casual(self):
        src = _p(material_family="cotton", shine_level="matte", sheen="matte")
        cand = _hoodie()
        assert _check_F1(src, cand) == 0.0

    def test_F2_satin_plus_hoodie(self):
        src = _p(material_family="satin")
        cand = _hoodie()
        assert _check_F2(src, cand) == pytest.approx(-0.10)

    def test_F2_silk_plus_fleece(self):
        src = _p(material_family="silk")
        cand = _p(gemini_category_l2="fleece")
        assert _check_F2(src, cand) == pytest.approx(-0.10)

    def test_F2_no_penalty_satin_plus_jeans(self):
        src = _p(material_family="satin")
        cand = _denim_jeans()
        assert _check_F2(src, cand) == 0.0

    def test_F3_distressed_plus_lingerie(self):
        src = _p(name="Distressed Boyfriend Jeans", texture="distressed")
        cand = _p(
            gemini_category_l2="camisole", material_family="silk",
        )
        assert _check_F3(src, cand) == pytest.approx(-0.08)

    def test_F3_no_penalty_plain_jeans_plus_camisole(self):
        src = _denim_jeans(name="Classic Jeans", texture="smooth")
        cand = _p(gemini_category_l2="camisole", material_family="silk")
        assert _check_F3(src, cand) == 0.0


class TestRuleG_OccasionMismatch:

    def test_G1_beach_plus_heavy_wool(self):
        src = _p(occasions=["beach", "vacation"])
        cand = _p(material_family="wool", fabric_weight="heavy")
        assert _check_G1(src, cand) == pytest.approx(-0.10)

    def test_G1_no_penalty_beach_plus_light_cotton(self):
        src = _p(occasions=["beach"])
        cand = _p(material_family="cotton", fabric_weight="light")
        assert _check_G1(src, cand) == 0.0

    def test_G2_office_plus_clubwear(self):
        src = _p(occasions=["office", "work"])
        cand = _p(coverage_level="revealing", formality_level=1)
        assert _check_G2(src, cand) == pytest.approx(-0.08)

    def test_G2_no_penalty_office_plus_moderate(self):
        src = _p(occasions=["office"])
        cand = _p(coverage_level="moderate", formality_level=2)
        assert _check_G2(src, cand) == 0.0

    def test_G3_formal_plus_denim_jacket(self):
        src = _p(formality_level=4)
        cand = _p(gemini_category_l2="denim jacket")
        assert _check_G3(src, cand) == pytest.approx(-0.10)

    def test_G3_no_penalty_casual_plus_denim_jacket(self):
        src = _p(formality_level=2)
        cand = _p(gemini_category_l2="denim jacket")
        assert _check_G3(src, cand) == 0.0


class TestRuleI_CategorySpecific:

    def test_I1_hoodie_plus_delicate(self):
        """Hoodie + silk shiny blouse → -0.10."""
        src = _hoodie()
        cand = _p(material_family="silk", shine_level="slight")
        assert _check_I1(src, cand) == pytest.approx(-0.10)

    def test_I1_no_penalty_hoodie_plus_cotton(self):
        src = _hoodie()
        cand = _p(material_family="cotton", shine_level="matte")
        assert _check_I1(src, cand) == 0.0

    def test_I2_blazer_plus_gym(self):
        """Blazer + track top → -0.10."""
        src = _blazer()
        cand = _p(gemini_category_l2="track top")
        assert _check_I2(src, cand) == pytest.approx(-0.10)

    def test_I2_blazer_plus_technical_sporty(self):
        """Blazer + technical fabric with sporty tag → -0.10."""
        src = _p(gemini_category_l2="blazer")
        cand = _p(material_family="technical", style_tags=["sporty"])
        assert _check_I2(src, cand) == pytest.approx(-0.10)

    def test_I3_formal_dress_plus_denim_jacket(self):
        src = _cocktail_dress()
        cand = _p(gemini_category_l2="denim jacket")
        assert _check_I3(src, cand) == pytest.approx(-0.10)

    def test_I3_no_penalty_midi_dress_plus_denim_jacket(self):
        """midi dress L2 is not in formal_dress_l2 set."""
        src = _p(gemini_category_l2="midi dress")
        cand = _p(gemini_category_l2="denim jacket")
        assert _check_I3(src, cand) == 0.0

    def test_I4_performance_legging_plus_formal_top(self):
        src = _performance_legging()
        cand = _p(formality_level=4, broad_category="tops")
        assert _check_I4(src, cand) == pytest.approx(-0.10)

    def test_I4_no_penalty_casual_leggings(self):
        """Cotton leggings (not synthetic_stretch) → no I4."""
        src = _p(
            gemini_category_l2="legging", material_family="cotton",
            broad_category="bottoms",
        )
        cand = _p(formality_level=4, broad_category="tops")
        assert _check_I4(src, cand) == 0.0


class TestRuleJ_ColorPrint:

    def test_J1_leopard_plus_zebra(self):
        src = _p(pattern="leopard")
        cand = _p(pattern="zebra")
        assert _check_J1(src, cand) == pytest.approx(-0.08)

    def test_J1_same_statement_print_no_penalty(self):
        """Same print is not 'competing' — just matchy."""
        src = _p(pattern="leopard")
        cand = _p(pattern="leopard")
        assert _check_J1(src, cand) == 0.0

    def test_J1_solid_plus_bold_no_penalty(self):
        src = _p(pattern="solid")
        cand = _p(pattern="leopard")
        assert _check_J1(src, cand) == 0.0

    def test_J2_neon_plus_neon(self):
        src = _p(color_saturation="bright", color_family="neon green")
        cand = _p(color_saturation="bright", color_family="neon pink", primary_color="neon pink")
        assert _check_J2(src, cand) == pytest.approx(-0.06)

    def test_J2_no_penalty_bright_but_not_neon(self):
        src = _p(color_saturation="bright", color_family="red")
        cand = _p(color_saturation="bright", color_family="blue")
        assert _check_J2(src, cand) == 0.0


# =========================================================================
# STYLE OVERRIDE TESTS
# =========================================================================

class TestStyleOverrides:

    def test_streetwear_reduces_E1_penalty(self):
        """Streetwear user: oversized+oversized penalty reduced to 30%."""
        src = _p(silhouette="oversized", broad_category="tops")
        cand = _p(silhouette="relaxed", broad_category="bottoms")
        penalty_default, _ = compute_avoid_penalties(src, cand)
        penalty_street, _ = compute_avoid_penalties(src, cand, user_styles={"streetwear"})
        # Default should be -0.08, streetwear should be -0.08 * 0.3 = -0.024
        assert penalty_default == pytest.approx(-0.08)
        assert penalty_street == pytest.approx(-0.08 * 0.3, abs=0.001)

    def test_monochrome_reduces_D1_penalty(self):
        """Monochrome user: denim-on-denim penalty reduced."""
        src = _denim_jeans()
        cand = _p(material_family="denim", gemini_category_l2="denim jacket")
        penalty_default, _ = compute_avoid_penalties(src, cand)
        penalty_mono, _ = compute_avoid_penalties(src, cand, user_styles={"monochrome"})
        assert penalty_default == pytest.approx(-0.10)
        assert penalty_mono == pytest.approx(-0.10 * 0.3, abs=0.001)

    def test_edgy_reduces_F3_penalty(self):
        """Edgy user: distressed+lingerie penalty reduced."""
        src = _p(name="Distressed Boyfriend Jeans", texture="distressed")
        cand = _p(gemini_category_l2="camisole", material_family="silk")
        penalty_default, rules_default = compute_avoid_penalties(src, cand)
        penalty_edgy, rules_edgy = compute_avoid_penalties(src, cand, user_styles={"edgy"})
        assert "F3" in rules_default
        assert "F3" in rules_edgy
        assert penalty_edgy > penalty_default  # less negative = higher

    def test_athleisure_overrides_hf3_hard_filter(self):
        """Athleisure user: blazer+track pants NOT hard-filtered."""
        source = _blazer()
        candidates = [_track_pants()]
        result = filter_hard_avoids(source, candidates, user_styles={"athleisure"})
        assert len(result) == 1

    def test_no_override_for_unknown_style(self):
        """Unknown style tag produces no override."""
        src = _denim_jeans()
        cand = _p(material_family="denim", gemini_category_l2="denim jacket")
        penalty_unknown, _ = compute_avoid_penalties(src, cand, user_styles={"cottagecore"})
        penalty_none, _ = compute_avoid_penalties(src, cand)
        assert penalty_unknown == penalty_none


# =========================================================================
# PENALTY CAP TEST
# =========================================================================

class TestPenaltyCap:

    def test_penalty_capped_at_minus_025(self):
        """Stack many penalties — total must not exceed PENALTY_CAP."""
        # Build an item that triggers many rules simultaneously:
        # formal evening (B1), party fabric (F1), shiny (F1),
        # different statement print (J1), neon (J2)
        src = _p(
            formality_level=5, material_family="sequin",
            shine_level="shiny", sheen="metallic",
            pattern="leopard", color_saturation="bright",
            color_family="neon green", primary_color="neon green",
            temp_band="hot", layer_role="base",
            occasions=["beach"],
        )
        cand = _p(
            formality_level=1, material_family="technical",
            gemini_category_l2="hoodie", fabric_weight="heavy",
            layer_role="midlayer", temp_band="cold",
            style_tags=["sporty"], pattern="zebra",
            color_saturation="bright", color_family="neon pink",
            primary_color="neon pink",
        )
        penalty, triggered = compute_avoid_penalties(src, cand)
        assert penalty >= PENALTY_CAP
        assert penalty == pytest.approx(PENALTY_CAP)
        # Should have triggered many rules
        assert len(triggered) >= 4


# =========================================================================
# GRACEFUL DEGRADATION TESTS
# =========================================================================

class TestGracefulDegradation:

    def test_missing_l2_no_crash(self):
        """None L2 on both sides → no crash, no penalty from L2-based rules."""
        src = _p(gemini_category_l2=None)
        cand = _p(gemini_category_l2=None)
        penalty, triggered = compute_avoid_penalties(src, cand)
        # No L2-based rules should fire
        l2_rules = {"C2", "F1", "F2", "I1", "I2", "I3", "I4", "G3"}
        assert not set(triggered) & l2_rules

    def test_missing_material_family_no_crash(self):
        src = _p(material_family=None)
        cand = _p(material_family=None)
        penalty, _ = compute_avoid_penalties(src, cand)
        assert penalty == 0.0 or penalty <= 0.0  # just verifying no crash

    def test_missing_formality_uses_default(self):
        """Missing formality_level defaults to 2 — no large gap."""
        src = _p()  # formality_level=2 (default)
        cand = _p()  # formality_level=2 (default)
        penalty, triggered = compute_avoid_penalties(src, cand)
        assert "B1" not in triggered

    def test_empty_occasions_no_crash(self):
        src = _p(occasions=[])
        cand = _p(occasions=[])
        penalty, triggered = compute_avoid_penalties(src, cand)
        assert "G1" not in triggered
        assert "G2" not in triggered

    def test_none_attributes_everywhere(self):
        """Maximally sparse profile — no crash."""
        sparse = _p(
            gemini_category_l2=None, material_family=None,
            fabric_weight=None, layer_role=None, temp_band=None,
            shine_level=None, sheen=None, silhouette=None, fit_type=None,
            pattern=None, color_family=None, primary_color=None,
            color_saturation=None, coverage_level=None, texture=None,
            occasions=[], style_tags=[], name=None, broad_category=None,
            length=None,
        )
        penalty, triggered = compute_avoid_penalties(sparse, sparse)
        assert isinstance(penalty, float)
        assert isinstance(triggered, list)


# =========================================================================
# COMPATIBLE PAIR TESTS (no penalty)
# =========================================================================

class TestCompatiblePairs:

    def test_casual_top_plus_jeans(self):
        """Classic casual combo — zero penalty."""
        penalty, triggered = compute_avoid_penalties(_casual_top(), _denim_jeans())
        assert penalty == 0.0
        assert triggered == []

    def test_fitted_top_plus_wide_bottom(self):
        """Good silhouette balance — zero penalty."""
        src = _p(silhouette="fitted", broad_category="tops", fabric_weight="light")
        cand = _wide_leg_pants()
        penalty, triggered = compute_avoid_penalties(src, cand)
        assert penalty == 0.0

    def test_blazer_plus_jeans(self):
        """Smart casual combo — zero penalty."""
        penalty, triggered = compute_avoid_penalties(_blazer(), _denim_jeans())
        assert penalty == 0.0

    def test_cocktail_dress_plus_structured_coat(self):
        """Formal dress + structured coat — compatible."""
        src = _cocktail_dress()
        cand = _p(
            gemini_category_l2="coat", formality_level=4,
            material_family="wool", fabric_weight="heavy",
            broad_category="outerwear", layer_role="outer",
        )
        penalty, triggered = compute_avoid_penalties(src, cand)
        assert penalty == 0.0


# =========================================================================
# INTEGRATION: compute_avoid_penalties triggered-rules output
# =========================================================================

class TestTriggeredRulesOutput:

    def test_triggered_rules_returned(self):
        """Verify triggered rule IDs are returned."""
        src = _hoodie()
        cand = _p(material_family="satin", shine_level="slight")
        _, triggered = compute_avoid_penalties(src, cand)
        assert "I1" in triggered

    def test_multiple_rules_can_stack(self):
        """Multiple independent rules can fire simultaneously."""
        # Hoodie (formality 1) + formal silk satin candidate (formality 5)
        src = _hoodie()
        cand = _p(
            formality_level=5, material_family="satin",
            shine_level="slight", sheen="satin",
            gemini_category_l2="formal blouse",
        )
        penalty, triggered = compute_avoid_penalties(src, cand)
        # Should trigger B1 (formality gap 4), I1 (hoodie + delicate),
        # F2 (satin + hoodie)
        assert "B1" in triggered
        assert "I1" in triggered
        assert "F2" in triggered
        assert penalty < -0.20  # stacked

    def test_no_rules_for_compatible(self):
        penalty, triggered = compute_avoid_penalties(_casual_top(), _denim_jeans())
        assert triggered == []
        assert penalty == 0.0


# =========================================================================
# K1: STYLE-TAG COHERENCE (refined vs sporty world clash)
# =========================================================================

class TestK1StyleCoherence:

    def test_refined_source_sporty_candidate(self):
        """Classic/Chic jeans + Sporty tracksuit top → penalty."""
        src = _p(style_tags=["Classic", "Casual", "Chic"])
        cand = _p(style_tags=["Sporty", "Casual", "Streetwear"])
        assert _check_K1(src, cand) == -0.06

    def test_sporty_source_refined_candidate(self):
        """Sporty joggers + Chic blouse → penalty (bidirectional)."""
        src = _p(style_tags=["Sporty", "Casual"])
        cand = _p(style_tags=["Classic", "Chic", "Elegant"])
        assert _check_K1(src, cand) == -0.06

    def test_bridge_item_no_penalty(self):
        """Candidate has both Sporty and Chic → bridge item, no penalty."""
        src = _p(style_tags=["Classic", "Chic"])
        cand = _p(style_tags=["Sporty", "Chic"])  # bridge
        assert _check_K1(src, cand) == 0.0

    def test_both_refined_no_penalty(self):
        """Both items refined → no clash."""
        src = _p(style_tags=["Classic", "Chic"])
        cand = _p(style_tags=["Elegant", "Minimalist"])
        assert _check_K1(src, cand) == 0.0

    def test_both_sporty_no_penalty(self):
        """Both items sporty → no clash."""
        src = _p(style_tags=["Sporty", "Streetwear"])
        cand = _p(style_tags=["Sporty", "Casual"])
        assert _check_K1(src, cand) == 0.0

    def test_neutral_only_no_penalty(self):
        """Only neutral tags (Casual, Trendy) → no clash."""
        src = _p(style_tags=["Casual", "Trendy"])
        cand = _p(style_tags=["Casual", "Modern"])
        assert _check_K1(src, cand) == 0.0

    def test_empty_styles_no_penalty(self):
        """Missing style tags → no crash, no penalty."""
        src = _p(style_tags=[])
        cand = _p(style_tags=["Sporty"])
        assert _check_K1(src, cand) == 0.0

    def test_none_styles_no_penalty(self):
        """None style tags → no crash, no penalty."""
        src = _p(style_tags=None)
        cand = _p(style_tags=["Classic", "Chic"])
        assert _check_K1(src, cand) == 0.0

    def test_case_insensitive(self):
        """Style tags are case-insensitive."""
        src = _p(style_tags=["CLASSIC", "Chic"])
        cand = _p(style_tags=["sporty", "STREETWEAR"])
        assert _check_K1(src, cand) == -0.06

    def test_romantic_vs_sporty(self):
        """Romantic (refined) + Sporty → penalty."""
        src = _p(style_tags=["Romantic", "Bohemian"])
        cand = _p(style_tags=["Sporty"])
        assert _check_K1(src, cand) == -0.06

    def test_glamorous_vs_streetwear(self):
        """Glamorous (refined) + Streetwear (sporty) → penalty."""
        src = _p(style_tags=["Glamorous", "Party"])
        cand = _p(style_tags=["Streetwear", "Trendy"])
        assert _check_K1(src, cand) == -0.06

    def test_minimalist_vs_sporty(self):
        """Minimalist (refined) + Sporty → penalty."""
        src = _p(style_tags=["Minimalist"])
        cand = _p(style_tags=["Sporty"])
        assert _check_K1(src, cand) == -0.06

    def test_edgy_is_neutral(self):
        """Edgy is neutral — doesn't count as refined or sporty."""
        src = _p(style_tags=["Edgy"])  # neutral only
        cand = _p(style_tags=["Sporty"])
        assert _check_K1(src, cand) == 0.0

    def test_bohemian_is_neutral(self):
        """Bohemian is neutral — doesn't count as refined or sporty."""
        src = _p(style_tags=["Bohemian"])  # neutral only
        cand = _p(style_tags=["Sporty"])
        assert _check_K1(src, cand) == 0.0

    def test_k1_in_compute_penalties(self):
        """K1 fires through the full compute_avoid_penalties pipeline."""
        src = _p(style_tags=["Classic", "Chic", "Casual"])
        cand = _p(style_tags=["Sporty", "Streetwear", "Casual"])
        penalty, triggered = compute_avoid_penalties(src, cand)
        assert "K1" in triggered
        assert penalty <= -0.06

    def test_k1_override_by_streetwear(self):
        """Streetwear user style reduces K1 to 30%."""
        src = _p(style_tags=["Classic", "Chic"])
        cand = _p(style_tags=["Sporty", "Streetwear"])
        penalty, triggered = compute_avoid_penalties(
            src, cand, user_styles={"streetwear"},
        )
        assert "K1" in triggered
        # -0.06 * 0.3 = -0.018
        assert -0.02 < penalty < 0.0

    def test_k1_override_by_athleisure(self):
        """Athleisure user style reduces K1 to 30%."""
        src = _p(style_tags=["Classic", "Elegant"])
        cand = _p(style_tags=["Sporty"])
        penalty, triggered = compute_avoid_penalties(
            src, cand, user_styles={"athleisure"},
        )
        assert "K1" in triggered
        assert -0.02 < penalty < 0.0


# ===========================================================================
# Lifestyle context classifiers
# ===========================================================================

class TestActiveContext:
    """Tests for _has_active_context() multi-signal detection."""

    def test_activewear_style_tag(self):
        """'activewear' style tag is a strong signal."""
        p = _p(style_tags=["activewear"])
        assert _has_active_context(p)

    def test_athletic_style_tag(self):
        """'athletic' style tag is a strong signal."""
        p = _p(style_tags=["athletic"])
        assert _has_active_context(p)

    def test_sporty_alone_is_not_active(self):
        """'sporty' style tag alone does NOT trigger active context."""
        p = _p(style_tags=["sporty"])
        assert not _has_active_context(p)

    def test_sporty_with_technical_material(self):
        """'sporty' + technical material = active context."""
        p = _p(style_tags=["sporty"], material_family="technical")
        assert _has_active_context(p)

    def test_sporty_with_synthetic_stretch(self):
        """'sporty' + synthetic_stretch = active context."""
        p = _p(style_tags=["sporty"], material_family="synthetic_stretch")
        assert _has_active_context(p)

    def test_sporty_with_cotton(self):
        """'sporty' + cotton does NOT trigger — just a casual sporty item."""
        p = _p(style_tags=["sporty"], material_family="cotton")
        assert not _has_active_context(p)

    def test_workout_occasion(self):
        """Workout occasion is a strong signal."""
        p = _p(occasions=["workout"])
        assert _has_active_context(p)

    def test_gym_occasion(self):
        p = _p(occasions=["gym"])
        assert _has_active_context(p)

    def test_yoga_occasion(self):
        p = _p(occasions=["yoga"])
        assert _has_active_context(p)

    def test_everyday_occasion_not_active(self):
        """Everyday is not an active context."""
        p = _p(occasions=["everyday"])
        assert not _has_active_context(p)

    def test_activewear_l2(self):
        """Performance legging L2 is a strong signal."""
        p = _p(gemini_category_l2="performance legging")
        assert _has_active_context(p)

    def test_sports_bra_l2(self):
        p = _p(gemini_category_l2="sports bra")
        assert _has_active_context(p)

    def test_regular_tank_not_active(self):
        """A regular tank top is NOT active context."""
        p = _p(gemini_category_l2="tank top")
        assert not _has_active_context(p)

    def test_name_keyword_active(self):
        """'active' in name triggers active context."""
        p = _p(name="Aspire Active Tank")
        assert _has_active_context(p)

    def test_name_keyword_performance(self):
        p = _p(name="Performance Crop Top")
        assert _has_active_context(p)

    def test_name_keyword_sport(self):
        p = _p(name="Sport Essentials Legging")
        assert _has_active_context(p)

    def test_name_keyword_training(self):
        p = _p(name="Training Tank Black")
        assert _has_active_context(p)

    def test_normal_name_not_active(self):
        """A regular product name is NOT active context."""
        p = _p(name="Ribbed Knit Cardigan")
        assert not _has_active_context(p)

    def test_classic_not_active(self):
        """Classic style tag is not active."""
        p = _p(style_tags=["classic", "minimalist"])
        assert not _has_active_context(p)

    def test_multiple_signals_combine(self):
        """Athletic tag + workout occasion = active (multiple signals)."""
        p = _p(style_tags=["athletic"], occasions=["workout"])
        assert _has_active_context(p)


class TestPolishedContext:
    """Tests for _has_polished_context() multi-signal detection."""

    def test_classic_style(self):
        p = _p(style_tags=["classic"])
        assert _has_polished_context(p)

    def test_minimalist_style(self):
        p = _p(style_tags=["minimalist"])
        assert _has_polished_context(p)

    def test_chic_style(self):
        p = _p(style_tags=["chic"])
        assert _has_polished_context(p)

    def test_elegant_style(self):
        p = _p(style_tags=["elegant"])
        assert _has_polished_context(p)

    def test_formal_style(self):
        p = _p(style_tags=["formal"])
        assert _has_polished_context(p)

    def test_high_formality_level(self):
        """Formality level >= 3 (business casual+) is polished."""
        p = _p(formality_level=3, style_tags=["casual"])
        assert _has_polished_context(p)

    def test_low_formality_not_polished(self):
        """Formality level 1 with no polished styles is not polished."""
        p = _p(formality_level=1, style_tags=["casual"])
        assert not _has_polished_context(p)

    def test_tailored_l2(self):
        """Blazer L2 is polished."""
        p = _p(gemini_category_l2="blazer", formality_level=2, style_tags=[])
        assert _has_polished_context(p)

    def test_office_occasion(self):
        p = _p(occasions=["office"], formality_level=2, style_tags=[])
        assert _has_polished_context(p)

    def test_work_occasion(self):
        p = _p(occasions=["work"], formality_level=2, style_tags=[])
        assert _has_polished_context(p)

    def test_sporty_not_polished(self):
        """Sporty style is not polished."""
        p = _p(style_tags=["sporty"], formality_level=1)
        assert not _has_polished_context(p)

    def test_bohemian_not_polished(self):
        """Bohemian is not polished."""
        p = _p(style_tags=["bohemian"], formality_level=2)
        assert not _has_polished_context(p)


# ===========================================================================
# L: Lifestyle / Context Mismatch
# ===========================================================================

class TestL1LifestyleMismatch:
    """Tests for L1: active candidate + polished source."""

    def test_aspire_tank_with_tailored_pants(self):
        """The exact scenario: athleisure tank + tailored wide-leg pant."""
        src = _p(
            name="Baccarat Mid Rise Wide Leg Belted Pant",
            style_tags=["classic", "minimalist"],
            formality_level=3,
            gemini_category_l2="tailored trousers",
            occasions=["work", "everyday"],
        )
        cand = _p(
            name="Aspire Active Tank",
            style_tags=["casual"],
            gemini_category_l2="tank top",
            occasions=["everyday"],
        )
        # Name contains "Active " → active context
        assert _check_L1(src, cand) == -0.15

    def test_activewear_tag_cand_polished_source(self):
        """Candidate with 'activewear' tag + classic source."""
        src = _p(style_tags=["classic", "elegant"], formality_level=3)
        cand = _p(style_tags=["activewear"])
        assert _check_L1(src, cand) == -0.15

    def test_gym_occasion_cand_polished_source(self):
        """Candidate with gym occasion + polished source."""
        src = _p(style_tags=["minimalist"], formality_level=3)
        cand = _p(occasions=["gym", "everyday"], style_tags=["casual"])
        assert _check_L1(src, cand) == -0.15

    def test_performance_name_cand_polished_source(self):
        """Candidate with 'performance' in name + polished source."""
        src = _p(style_tags=["classic"])
        cand = _p(name="Performance Crop Top", style_tags=["casual"])
        assert _check_L1(src, cand) == -0.15

    def test_reverse_direction(self):
        """Active source + polished candidate → -0.12 (smaller penalty)."""
        src = _p(style_tags=["activewear"], occasions=["workout"])
        cand = _p(style_tags=["classic", "elegant"], formality_level=4)
        assert _check_L1(src, cand) == -0.12

    def test_bridge_item_no_penalty(self):
        """Candidate with BOTH active AND polished signals → no penalty."""
        src = _p(style_tags=["classic"], formality_level=3)
        cand = _p(
            style_tags=["activewear", "chic"],  # bridge: both worlds
            occasions=["workout", "office"],
        )
        # Candidate has polished context (chic + office), so bridge check passes
        assert _check_L1(src, cand) == 0.0

    def test_both_polished_no_penalty(self):
        """Two polished items → no penalty."""
        src = _p(style_tags=["classic"], formality_level=3)
        cand = _p(style_tags=["elegant"], formality_level=4)
        assert _check_L1(src, cand) == 0.0

    def test_both_active_no_penalty(self):
        """Two active items → no penalty."""
        src = _p(style_tags=["activewear"], occasions=["gym"])
        cand = _p(style_tags=["athletic"], occasions=["workout"])
        assert _check_L1(src, cand) == 0.0

    def test_casual_vs_polished_no_penalty(self):
        """Plain casual item (not active) vs polished → no penalty."""
        src = _p(style_tags=["classic"], formality_level=3)
        cand = _p(style_tags=["casual"], formality_level=1)
        assert _check_L1(src, cand) == 0.0

    def test_sporty_alone_no_penalty(self):
        """Sporty-only cand vs polished source → no L1 (K1 handles it)."""
        src = _p(style_tags=["classic"], formality_level=3)
        cand = _p(style_tags=["sporty"])
        assert _check_L1(src, cand) == 0.0

    def test_l1_in_pipeline(self):
        """L1 fires through compute_avoid_penalties."""
        src = _p(style_tags=["classic", "minimalist"], formality_level=3)
        cand = _p(style_tags=["activewear"], occasions=["workout"])
        penalty, triggered = compute_avoid_penalties(src, cand)
        assert "L1" in triggered
        assert penalty <= -0.15

    def test_l1_override_by_athleisure(self):
        """Athleisure user reduces L1 to 30%."""
        src = _p(style_tags=["classic"], formality_level=3)
        cand = _p(style_tags=["activewear"])
        penalty, triggered = compute_avoid_penalties(
            src, cand, user_styles={"athleisure"},
        )
        assert "L1" in triggered
        # -0.15 * 0.3 = -0.045
        assert -0.05 < penalty < 0.0

    def test_l1_override_by_sporty_chic(self):
        """Sporty chic user reduces L1 to 30%."""
        src = _p(style_tags=["elegant"], formality_level=4)
        cand = _p(style_tags=["athletic"], occasions=["gym"])
        penalty, triggered = compute_avoid_penalties(
            src, cand, user_styles={"sporty chic"},
        )
        assert "L1" in triggered
        # Check penalty is reduced (not full -0.15)
        assert penalty > -0.10


class TestL2ActiveVsEvening:
    """Tests for L2: active item paired with evening/dressy item."""

    def test_activewear_cand_evening_source(self):
        """Activewear candidate + evening source."""
        src = _p(
            formality_level=4, occasions=["formal event"],
            style_tags=["glamorous"], gemini_category_l2="cocktail dress",
        )
        cand = _p(style_tags=["activewear"], occasions=["workout"])
        assert _check_L2(src, cand) == -0.18

    def test_gym_cand_gala_source(self):
        """Gym-occasion candidate + gala-occasion source."""
        src = _p(occasions=["gala"], formality_level=5)
        cand = _p(occasions=["gym"], style_tags=["casual"])
        assert _check_L2(src, cand) == -0.18

    def test_reverse_active_src_evening_cand(self):
        """Active source + evening candidate → -0.15."""
        src = _p(style_tags=["activewear"], occasions=["workout"])
        cand = _p(
            formality_level=4, occasions=["formal event"],
            style_tags=["glamorous"],
        )
        assert _check_L2(src, cand) == -0.15

    def test_formal_evening_l2(self):
        """Evening dress L2 triggers evening context."""
        src = _p(gemini_category_l2="evening dress", formality_level=4)
        cand = _p(style_tags=["athletic"])
        assert _check_L2(src, cand) == -0.18

    def test_casual_vs_evening_no_l2(self):
        """Casual (not active) vs evening → no L2 penalty."""
        src = _p(formality_level=4, occasions=["formal event"])
        cand = _p(style_tags=["casual"], formality_level=1)
        assert _check_L2(src, cand) == 0.0

    def test_both_evening_no_penalty(self):
        """Two evening items → no penalty."""
        src = _p(formality_level=4, occasions=["formal event"])
        cand = _p(formality_level=4, occasions=["gala"])
        assert _check_L2(src, cand) == 0.0

    def test_l2_in_pipeline(self):
        """L2 fires through compute_avoid_penalties."""
        src = _p(
            formality_level=5, occasions=["gala"],
            gemini_category_l2="gown",
        )
        cand = _p(style_tags=["activewear"], occasions=["gym"])
        penalty, triggered = compute_avoid_penalties(src, cand)
        assert "L2" in triggered
        assert penalty <= -0.18

    def test_l2_override_by_athleisure(self):
        """Athleisure user reduces L2 to 30%."""
        src = _p(formality_level=4, occasions=["formal event"])
        cand = _p(style_tags=["activewear"])
        penalty, triggered = compute_avoid_penalties(
            src, cand, user_styles={"athleisure"},
        )
        assert "L2" in triggered
        # -0.18 * 0.3 = -0.054
        assert penalty > -0.10

    def test_sporty_alone_no_l2(self):
        """Sporty-only cand vs evening source → no L2 (K1 handles it)."""
        src = _p(formality_level=4, occasions=["formal event"])
        cand = _p(style_tags=["sporty"])
        assert _check_L2(src, cand) == 0.0
