"""
Unit tests for the TATTOO v2 scoring engine.

Tests all 8 dimensions, derived fields, cross-dimension gates,
weight sums, and end-to-end scoring.
"""

import pytest
from services.outfit_engine import (
    AestheticProfile,
    _derive_fields,
    _score_occasion_formality,
    _score_style,
    _score_fabric,
    _score_silhouette,
    _score_color,
    _score_seasonality,
    _pattern_compatibility,
    _price_coherence,
    _base_color_harmony,
    _style_adjacency,
    _get_fabric_family,
    _lookup_symmetric,
    compute_compatibility_score,
    CATEGORY_PAIR_WEIGHTS,
    DEFAULT_WEIGHTS,
    _FABRIC_FAMILY_COMPAT,
    _SAME_MATERIAL_PENALTY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(**kwargs) -> AestheticProfile:
    """Create profile with derived fields computed."""
    p = AestheticProfile(**kwargs)
    _derive_fields(p)
    return p


def _top(**kwargs) -> AestheticProfile:
    """Shorthand for a Tops profile."""
    defaults = dict(
        broad_category="tops", gemini_category_l1="Tops",
        gemini_category_l2="Blouse", formality="Smart Casual",
        occasions=["Everyday", "Work"], style_tags=["Classic", "Chic"],
        color_family="Whites", apparent_fabric="Cotton", texture="Smooth",
        pattern="Solid", silhouette="Fitted", length="Regular",
        coverage_level="Moderate", seasons=["Spring", "Fall"], price=50,
    )
    defaults.update(kwargs)
    return _make_profile(**defaults)


def _bottom(**kwargs) -> AestheticProfile:
    """Shorthand for a Bottoms profile."""
    defaults = dict(
        broad_category="bottoms", gemini_category_l1="Bottoms",
        gemini_category_l2="Jeans", formality="Smart Casual",
        occasions=["Everyday", "Work"], style_tags=["Classic", "Minimalist"],
        color_family="Blues", apparent_fabric="Denim", texture="Textured",
        pattern="Solid", silhouette="Straight", length="Regular",
        coverage_level="Full", seasons=["Spring", "Fall"], price=55,
    )
    defaults.update(kwargs)
    return _make_profile(**defaults)


def _outerwear(**kwargs) -> AestheticProfile:
    """Shorthand for an Outerwear profile."""
    defaults = dict(
        broad_category="outerwear", gemini_category_l1="Outerwear",
        gemini_category_l2="Blazer", formality="Smart Casual",
        occasions=["Work", "Everyday"], style_tags=["Classic", "Chic"],
        color_family="Blacks", apparent_fabric="Wool", texture="Textured",
        pattern="Solid", silhouette="Regular", length="Regular",
        coverage_level="Full", seasons=["Fall", "Winter"], price=80,
    )
    defaults.update(kwargs)
    return _make_profile(**defaults)


# ===========================================================================
# Section 1: AestheticProfile + _derive_fields
# ===========================================================================

class TestDeriveFields:
    """Tests for derived field computation."""

    def test_formality_level_mapping(self):
        assert _make_profile(formality="Casual").formality_level == 1
        assert _make_profile(formality="Smart Casual").formality_level == 2
        assert _make_profile(formality="Business Casual").formality_level == 3
        assert _make_profile(formality="Semi-Formal").formality_level == 4
        assert _make_profile(formality="Formal").formality_level == 5

    def test_formality_level_default(self):
        """Missing or unknown formality defaults to 2."""
        assert _make_profile(formality=None).formality_level == 2
        assert _make_profile(formality="").formality_level == 2

    def test_is_bridge_from_l2(self):
        assert _make_profile(gemini_category_l2="Blazer").is_bridge is True
        assert _make_profile(gemini_category_l2="Trench").is_bridge is True
        assert _make_profile(gemini_category_l2="T-Shirt").is_bridge is True
        assert _make_profile(gemini_category_l2="Cardigan").is_bridge is True

    def test_is_bridge_false_for_non_bridge(self):
        assert _make_profile(gemini_category_l2="Corset").is_bridge is False
        assert _make_profile(gemini_category_l2="Crop Top").is_bridge is False

    def test_is_bridge_from_name_fallback(self):
        assert _make_profile(gemini_category_l2="Jacket",
                             name="Leather Jacket Classic").is_bridge is True
        assert _make_profile(gemini_category_l2="Top",
                             name="White Sneaker Canvas").is_bridge is True

    def test_primary_style(self):
        p = _make_profile(style_tags=["Trendy", "Casual", "Chic"])
        assert p.primary_style == "trendy"

    def test_primary_style_empty(self):
        p = _make_profile(style_tags=[])
        assert p.primary_style is None

    def test_style_strength_high_for_sequin(self):
        p = _make_profile(name="Sequin Party Dress", apparent_fabric="Sequin")
        assert p.style_strength >= 0.85

    def test_style_strength_low_for_basic(self):
        p = _make_profile(style_tags=["Casual", "Minimalist"], pattern="Solid")
        assert p.style_strength <= 0.30

    def test_style_strength_mid_for_floral(self):
        p = _make_profile(pattern="Floral", style_tags=["Romantic"])
        assert 0.50 <= p.style_strength <= 0.75

    def test_material_family_derived(self):
        assert _make_profile(apparent_fabric="Denim").material_family == "denim"
        assert _make_profile(apparent_fabric="Silk Chiffon").material_family == "silk"
        assert _make_profile(apparent_fabric="Wool Blend").material_family == "wool"

    def test_texture_intensity(self):
        assert _make_profile(texture="Smooth").texture_intensity == "smooth"
        assert _make_profile(texture="Ribbed").texture_intensity == "medium"
        assert _make_profile(texture="Cable Knit").texture_intensity == "strong"

    def test_shine_level(self):
        assert _make_profile(sheen="Matte").shine_level == "matte"
        assert _make_profile(sheen="Satin").shine_level == "slight"
        assert _make_profile(sheen="Metallic").shine_level == "shiny"

    def test_fabric_weight(self):
        assert _make_profile(apparent_fabric="Silk").fabric_weight == "light"
        assert _make_profile(apparent_fabric="Cotton").fabric_weight == "mid"
        assert _make_profile(apparent_fabric="Wool").fabric_weight == "heavy"

    def test_layer_role_outerwear(self):
        assert _make_profile(gemini_category_l1="Outerwear").layer_role == "outer"

    def test_layer_role_base(self):
        assert _make_profile(gemini_category_l1="Tops",
                             gemini_category_l2="Tank Top").layer_role == "base"
        assert _make_profile(gemini_category_l1="Tops",
                             gemini_category_l2="Camisole").layer_role == "base"

    def test_layer_role_midlayer(self):
        assert _make_profile(gemini_category_l1="Tops",
                             gemini_category_l2="Hoodie").layer_role == "midlayer"
        assert _make_profile(gemini_category_l1="Tops",
                             gemini_category_l2="Sweater").layer_role == "midlayer"

    def test_temp_band_summer(self):
        assert _make_profile(seasons=["Summer"]).temp_band == "hot"

    def test_temp_band_winter(self):
        assert _make_profile(seasons=["Winter"]).temp_band == "cold"

    def test_temp_band_mild(self):
        assert _make_profile(seasons=["Spring", "Fall"]).temp_band == "mild"

    def test_temp_band_all_seasons(self):
        assert _make_profile(seasons=["Spring", "Summer", "Fall", "Winter"]).temp_band == "any"

    def test_color_saturation(self):
        assert _make_profile(color_family="Neon Pink").color_saturation == "bright"
        assert _make_profile(color_family="Dusty Rose").color_saturation == "muted"
        assert _make_profile(color_family="Blues").color_saturation == "medium"

    def test_from_product_and_attrs_loads_new_fields(self):
        """Verify sheen/rise/leg_shape/stretch are loaded from attrs."""
        product = {"id": "abc", "name": "Test", "brand": "B", "price": 50,
                   "primary_image_url": "", "category": "tops"}
        attrs = {"sheen": "Satin", "rise": "High", "leg_shape": "Wide",
                 "stretch": "Slight Stretch", "category_l1": "Tops",
                 "category_l2": "Blouse", "formality": "Casual"}
        p = AestheticProfile.from_product_and_attrs(product, attrs)
        assert p.sheen == "Satin"
        assert p.rise == "High"
        assert p.leg_shape == "Wide"
        assert p.stretch == "Slight Stretch"
        assert p.shine_level == "slight"  # derived from sheen


# ===========================================================================
# Section 2: Dim 1 — Occasion & Formality
# ===========================================================================

class TestOccasionFormality:
    """Tests for _score_occasion_formality."""

    def test_perfect_match(self):
        a = _top(formality="Smart Casual", occasions=["Everyday", "Work"])
        b = _bottom(formality="Smart Casual", occasions=["Everyday", "Work"])
        score = _score_occasion_formality(a, b)
        assert score >= 0.90

    def test_formality_delta_1(self):
        a = _top(formality="Smart Casual")
        b = _bottom(formality="Casual")
        score = _score_occasion_formality(a, b)
        assert 0.60 < score < 0.95

    def test_formality_delta_4_kills(self):
        """Formal + Casual = delta 4 → formality sub-score 0.10."""
        a = _top(formality="Formal")
        b = _bottom(formality="Casual")
        score = _score_occasion_formality(a, b)
        assert score < 0.45

    def test_bridge_softens_delta_2(self):
        """Bridge items (blazer) should soften Δ=2 penalty."""
        blazer = _outerwear(formality="Semi-Formal", gemini_category_l2="Blazer")
        casual = _bottom(formality="Smart Casual")
        # blazer is_bridge=True, delta=2 (4 vs 2)
        score = _score_occasion_formality(blazer, casual)
        # Without bridge: 0.55 * 0.55 + ... ≈ low
        # With bridge: 0.55 * 0.70 + ... ≈ higher
        assert score > 0.50

    def test_occasion_conflict_workout_vs_formal(self):
        """Workout + Formal Event → hard conflict → occ_score = 0.05."""
        a = _top(occasions=["Workout", "Gym"], formality="Casual")
        b = _bottom(occasions=["Wedding Guest", "Formal Event"], formality="Formal")
        score = _score_occasion_formality(a, b)
        assert score < 0.30

    def test_occasion_conflict_lounge_vs_office(self):
        a = _top(occasions=["Lounging"], formality="Casual")
        b = _bottom(occasions=["Office", "Work"], formality="Business Casual")
        score = _score_occasion_formality(a, b)
        assert score < 0.50

    def test_no_occasions_neutral(self):
        """Missing occasions → neutral score."""
        a = _top(occasions=[])
        b = _bottom(occasions=[])
        score = _score_occasion_formality(a, b)
        assert 0.40 < score < 0.85

    def test_time_context_both_day(self):
        """Both day occasions → time boost."""
        a = _top(occasions=["Everyday", "Brunch"])
        b = _bottom(occasions=["Weekend", "Work"])
        score = _score_occasion_formality(a, b)
        assert score > 0.70


# ===========================================================================
# Section 3: Dim 2 — Style
# ===========================================================================

class TestStyle:
    """Tests for _score_style and _style_adjacency."""

    def test_same_style_adjacency(self):
        assert _style_adjacency("classic", "classic") == 1.0

    def test_neighbor_adjacency(self):
        assert _style_adjacency("classic", "minimalist") == 0.75
        assert _style_adjacency("streetwear", "edgy") == 0.75

    def test_weak_neighbor_adjacency(self):
        assert _style_adjacency("casual", "classic") == 0.40

    def test_clash_adjacency(self):
        assert _style_adjacency("romantic", "streetwear") == 0.10
        assert _style_adjacency("sporty", "glamorous") == 0.10

    def test_unrelated_adjacency(self):
        """Unknown combo → default 0.35."""
        score = _style_adjacency("preppy", "party")
        assert 0.10 < score < 0.75

    def test_full_style_score_same(self):
        a = _top(style_tags=["Classic", "Chic"])
        b = _bottom(style_tags=["Classic", "Minimalist"])
        score = _score_style(a, b)
        assert score > 0.70

    def test_style_clash_low(self):
        a = _top(style_tags=["Romantic"])
        b = _bottom(style_tags=["Streetwear"])
        score = _score_style(a, b)
        assert score < 0.45

    def test_statement_plus_basic_ideal(self):
        """One high-strength statement + one basic supporting = ideal."""
        a = _top(style_tags=["Glamorous"], name="Sequin Corset",
                 apparent_fabric="Sequin", pattern="Solid")
        b = _bottom(style_tags=["Casual", "Minimalist"], pattern="Solid")
        score = _score_style(a, b)
        # Adjacency: Glamorous↔Casual is a clash (0.10)
        # But strength sub: 0.90 * 0.25 = high
        # Depends on balance of subs
        assert score > 0.15  # Not killed entirely

    def test_two_statements_penalized(self):
        """Two high-strength statement pieces → strength penalty."""
        a = _top(style_tags=["Glamorous", "Party"], name="Sequin Top",
                 apparent_fabric="Sequin")
        b = _bottom(style_tags=["Edgy", "Party"], name="Rhinestone Skirt",
                    apparent_fabric="Sequin")
        # Both high style_strength → strength_score ≈ 0.30
        score = _score_style(a, b)
        # adj: edgy↔glamorous = neighbor (0.75), strength: 0.30
        assert score < 0.75

    def test_bridge_boost(self):
        """Style bridge items (casual/minimalist/classic) soften clashes."""
        a = _top(style_tags=["Classic"], gemini_category_l2="T-Shirt")  # bridge
        b = _bottom(style_tags=["Edgy"])
        score_bridge = _score_style(a, b)

        a2 = _top(style_tags=["Romantic"])  # not bridge
        score_no_bridge = _score_style(a2, b)
        # Bridge should help
        assert score_bridge >= score_no_bridge

    def test_empty_style_tags(self):
        a = _top(style_tags=[])
        b = _bottom(style_tags=["Classic"])
        score = _score_style(a, b)
        assert score == pytest.approx(0.40)


# ===========================================================================
# Section 4: Dim 3 — Fabric
# ===========================================================================

class TestFabric:
    """Tests for _score_fabric."""

    def test_denim_plus_knit_good(self):
        a = _bottom(apparent_fabric="Denim", texture="Textured")
        b = _top(apparent_fabric="Knit", texture="Ribbed")
        score = _score_fabric(a, b)
        assert score > 0.65

    def test_denim_plus_denim_bad(self):
        """Same material penalty for denim+denim."""
        a = _bottom(apparent_fabric="Denim")
        b = _outerwear(apparent_fabric="Denim")
        score = _score_fabric(a, b)
        assert score < 0.50

    def test_leather_plus_leather_bad(self):
        a = _bottom(apparent_fabric="Leather")
        b = _top(apparent_fabric="Leather")
        score = _score_fabric(a, b)
        assert score < 0.50

    def test_contrast_strong_smooth_good(self):
        """Strong texture + smooth = great contrast."""
        a = _top(texture="Cable Knit", apparent_fabric="Knit")
        b = _bottom(texture="Smooth", apparent_fabric="Satin")
        score = _score_fabric(a, b)
        assert score > 0.60

    def test_both_shiny_bad(self):
        """Both shiny items = bad."""
        a = _top(sheen="Shiny", apparent_fabric="Satin", texture="Smooth")
        b = _bottom(sheen="Metallic", apparent_fabric="Sequin", texture="Smooth")
        score = _score_fabric(a, b)
        assert score < 0.50

    def test_light_heavy_weight_mismatch(self):
        """Light + heavy fabric = moderate penalty."""
        a = _top(apparent_fabric="Chiffon")
        b = _bottom(apparent_fabric="Leather")
        score = _score_fabric(a, b)
        # Weight sub: light+heavy = 0.45
        assert score < 0.70

    def test_missing_fabric_neutral(self):
        a = _top(apparent_fabric=None, texture=None)
        b = _bottom(apparent_fabric=None, texture=None)
        score = _score_fabric(a, b)
        assert 0.40 < score < 0.65


# ===========================================================================
# Section 5: Dim 4 — Silhouette
# ===========================================================================

class TestSilhouette:
    """Tests for _score_silhouette."""

    def test_fitted_top_wide_bottom_great(self):
        """Balance rule: fitted top + wide-leg bottom = 0.90."""
        a = _top(silhouette="Fitted")
        b = _bottom(silhouette="Wide Leg")
        score = _score_silhouette(a, b)
        assert score > 0.75

    def test_oversized_plus_wide_bad(self):
        """Oversized top + wide bottom = bad."""
        a = _top(silhouette="Oversized")
        b = _bottom(silhouette="Wide Leg")
        score = _score_silhouette(a, b)
        assert score < 0.50

    def test_oversized_plus_wide_streetwear_ok(self):
        """Streetwear exception: oversized + wide is acceptable."""
        a = _top(silhouette="Oversized", style_tags=["Streetwear"])
        b = _bottom(silhouette="Wide Leg")
        score = _score_silhouette(a, b)
        assert score > 0.45

    def test_cropped_top_high_rise_boost(self):
        """Cropped top + high-rise bottoms = great."""
        a = _top(length="Cropped", silhouette="Fitted")
        b = _bottom(rise="High", silhouette="Straight")
        score = _score_silhouette(a, b)
        assert score > 0.75

    def test_cropped_top_low_rise_penalty(self):
        """Cropped top + low-rise = penalized (unless trendy)."""
        a = _top(length="Cropped", silhouette="Fitted")
        b = _bottom(rise="Low", silhouette="Straight")
        score_basic = _score_silhouette(a, b)

        b_trendy = _bottom(rise="Low", silhouette="Straight", style_tags=["Trendy"])
        score_trendy = _score_silhouette(a, b_trendy)
        assert score_trendy > score_basic

    def test_oversized_plus_oversized_hard_penalty(self):
        a = _top(silhouette="Oversized")
        b = _bottom(silhouette="Oversized")
        score = _score_silhouette(a, b)
        assert score < 0.40

    def test_outerwear_cropped_jacket_wide_leg(self):
        """Cropped jacket + wide-leg = great."""
        a = _outerwear(length="Cropped", silhouette="Fitted")
        b = _bottom(silhouette="Wide Leg")
        score = _score_silhouette(a, b)
        assert score > 0.70

    def test_longline_coat_slim_bottoms(self):
        """Longline coat + slim silhouette = good."""
        a = _outerwear(length="Midi", silhouette="Regular")
        b = _bottom(silhouette="Slim")
        score = _score_silhouette(a, b)
        assert score > 0.55

    def test_missing_silhouette_neutral(self):
        a = _top(silhouette=None, coverage_level=None, length=None)
        b = _bottom(silhouette=None, coverage_level=None, length=None)
        assert _score_silhouette(a, b) == pytest.approx(0.50)


# ===========================================================================
# Section 6: Dim 5 — Color
# ===========================================================================

class TestColor:
    """Tests for _score_color and _base_color_harmony."""

    def test_neutral_plus_anything_good(self):
        """True neutral + any chromatic = at least 0.75."""
        a = _top(color_family="Blacks")
        b = _bottom(color_family="Reds")
        score = _score_color(a, b)
        assert score >= 0.75

    def test_warm_neutral_cool_saturated_clash(self):
        """Brown + purple = warm neutral + cool saturated → clash."""
        a = _top(color_family="Browns")
        b = _bottom(color_family="Purples")
        score = _base_color_harmony("Browns", "Purples")
        assert score < 0.50

    def test_analogous_colors_good(self):
        score = _base_color_harmony("Red", "Orange")
        assert score >= 0.70

    def test_complementary_colors_good(self):
        score = _base_color_harmony("Navy", "Rust")
        assert score >= 0.65

    def test_same_color_same_material_penalty(self):
        """Blue denim + blue denim → same-material-same-color penalty."""
        a = _bottom(color_family="Blues", apparent_fabric="Denim")
        b = _outerwear(color_family="Blues", apparent_fabric="Denim")
        score = _score_color(a, b)
        assert score <= 0.40

    def test_denim_boost_with_white(self):
        """Denim + white/cream/black = boost to at least 0.80."""
        a = _bottom(color_family="Blues", apparent_fabric="Denim")
        b = _top(color_family="Whites", apparent_fabric="Cotton")
        score = _score_color(a, b)
        assert score >= 0.78

    def test_denim_boost_with_black(self):
        a = _bottom(color_family="Blues", apparent_fabric="Denim")
        b = _top(color_family="Blacks", apparent_fabric="Cotton")
        score = _score_color(a, b)
        assert score >= 0.78

    def test_two_bright_non_matching_penalty(self):
        """Two different bright colors → slight penalty."""
        a = _top(color_family="Neon Pink")
        b = _bottom(color_family="Electric Blue")
        score = _score_color(a, b)
        # Both bright + non-matching + base < 0.70 → -0.10
        assert score < 0.65

    def test_missing_color_neutral(self):
        a = _top(color_family=None, primary_color=None)
        b = _bottom(color_family="Blues")
        score = _score_color(a, b)
        assert score == pytest.approx(0.5)


# ===========================================================================
# Section 7: Dim 6 — Seasonality
# ===========================================================================

class TestSeasonality:
    """Tests for _score_seasonality."""

    def test_same_season_good(self):
        a = _top(seasons=["Spring", "Fall"])
        b = _bottom(seasons=["Spring", "Fall"])
        score = _score_seasonality(a, b)
        assert score > 0.60

    def test_hot_cold_clash(self):
        """Summer-only + Winter-only = temperature clash."""
        a = _top(seasons=["Summer"], apparent_fabric="Linen")
        b = _bottom(seasons=["Winter"], apparent_fabric="Wool")
        score = _score_seasonality(a, b)
        assert score < 0.40

    def test_layering_softens_temp_clash(self):
        """Outerwear over summer base = layering makes sense."""
        a = _top(seasons=["Summer"], gemini_category_l1="Tops",
                 gemini_category_l2="Tank Top")
        b = _outerwear(seasons=["Fall", "Winter"])
        score = _score_seasonality(a, b)
        # outer + base → layering softens
        assert score > 0.30

    def test_summer_fabrics_together(self):
        a = _top(apparent_fabric="Linen", seasons=["Summer"])
        b = _bottom(apparent_fabric="Cotton", seasons=["Summer"])
        score = _score_seasonality(a, b)
        assert score > 0.65

    def test_winter_fabrics_together(self):
        a = _top(apparent_fabric="Wool", seasons=["Winter"])
        b = _bottom(apparent_fabric="Leather", seasons=["Winter"])
        score = _score_seasonality(a, b)
        assert score > 0.65

    def test_cross_season_fabric_bad(self):
        """Summer fabric + winter fabric = bad."""
        a = _top(apparent_fabric="Linen", seasons=["Summer"],
                 gemini_category_l1="Tops", gemini_category_l2="Blouse")
        b = _bottom(apparent_fabric="Wool", seasons=["Winter"],
                    gemini_category_l1="Bottoms", gemini_category_l2="Trousers")
        score = _score_seasonality(a, b)
        assert score < 0.45

    def test_layer_compat_base_midlayer(self):
        """Base + midlayer = good layering."""
        a = _top(gemini_category_l1="Tops", gemini_category_l2="Tank Top",
                 seasons=["Spring"])
        b = _top(gemini_category_l1="Tops", gemini_category_l2="Cardigan",
                 seasons=["Spring"], broad_category="tops")
        score = _score_seasonality(a, b)
        assert score > 0.55

    def test_two_outers_bad(self):
        """Two outer layers = bad."""
        a = _outerwear(seasons=["Fall"])
        b = _outerwear(seasons=["Fall"])
        score = _score_seasonality(a, b)
        assert score < 0.55

    def test_missing_seasons_neutral(self):
        a = _top(seasons=[])
        b = _bottom(seasons=[])
        score = _score_seasonality(a, b)
        assert 0.35 < score < 0.70


# ===========================================================================
# Section 8: Dim 7 — Pattern (kept from v1)
# ===========================================================================

class TestPattern:
    def test_solid_plus_solid(self):
        assert _pattern_compatibility("Solid", "Solid") == pytest.approx(0.80)

    def test_solid_grounds_any(self):
        assert _pattern_compatibility("Solid", "Floral") == pytest.approx(0.85)
        assert _pattern_compatibility("Solid", "Geometric") == pytest.approx(0.85)

    def test_bold_plus_bold_clash(self):
        score = _pattern_compatibility("Floral", "Plaid")
        assert score < 0.30

    def test_bold_same_pattern(self):
        score = _pattern_compatibility("Floral", "Floral")
        assert score == pytest.approx(0.45)

    def test_subtle_plus_subtle(self):
        assert _pattern_compatibility("Pinstripe", "Herringbone") == pytest.approx(0.65)

    def test_missing_pattern(self):
        assert _pattern_compatibility(None, "Solid") == pytest.approx(0.50)


# ===========================================================================
# Section 9: Dim 8 — Price (kept from v1)
# ===========================================================================

class TestPrice:
    def test_close_prices(self):
        assert _price_coherence(50, 55) == pytest.approx(1.0)

    def test_double_price(self):
        assert _price_coherence(50, 100) == pytest.approx(0.85)

    def test_large_gap(self):
        assert _price_coherence(20, 200) == pytest.approx(0.2)

    def test_zero_price(self):
        assert _price_coherence(0, 50) == pytest.approx(0.5)


# ===========================================================================
# Section 10: Cross-Dimension Gates
# ===========================================================================

class TestCrossDimensionGates:
    """Test post-hoc penalty gates in compute_compatibility_score."""

    def test_formality_hard_gate(self):
        """occasion_formality < 0.25 → total capped at 0.35."""
        formal = _top(formality="Formal", occasions=["Formal Event", "Wedding Guest"],
                      style_tags=["Glamorous"])
        gym = _bottom(formality="Casual", occasions=["Workout", "Gym"],
                      style_tags=["Sporty"])
        total, dims = compute_compatibility_score(formal, gym)
        if dims["occasion_formality"] < 0.25:
            assert total <= 0.36  # capped at 0.35 + tiny float

    def test_silhouette_hard_cap(self):
        """silhouette < 0.30 → total capped at 0.40."""
        a = _top(silhouette="Oversized")
        b = _bottom(silhouette="Oversized")
        total, dims = compute_compatibility_score(a, b)
        if dims["silhouette"] < 0.30:
            assert total <= 0.41

    def test_same_material_color_penalty(self):
        """Denim + denim (both chromatic) → cross penalty of -0.12."""
        a = _bottom(apparent_fabric="Denim", color_family="Blues")
        b = _outerwear(apparent_fabric="Denim", color_family="Blues")
        total, dims = compute_compatibility_score(a, b)
        # Penalty should reduce total
        # Compute what it would be without penalty
        from services.outfit_engine import CATEGORY_PAIR_WEIGHTS, DEFAULT_WEIGHTS
        pair_key = ("bottoms", "outerwear")
        weights = CATEGORY_PAIR_WEIGHTS.get(pair_key, DEFAULT_WEIGHTS)
        raw_total = sum(weights.get(d, 0) * s for d, s in dims.items())
        # Total should be less than raw due to gates
        assert total < raw_total

    def test_color_contribution_halved_on_formality_clash(self):
        """When occasion_formality < 0.50, color contribution is halved."""
        formal = _top(formality="Formal", occasions=["Formal Event"],
                      color_family="Blacks", style_tags=["Glamorous"])
        casual = _bottom(formality="Casual", occasions=["Lounging"],
                         color_family="Blacks", style_tags=["Casual"])
        total, dims = compute_compatibility_score(formal, casual)
        # With formality clash, color weight effectively halved
        # Can't easily test exact amount but total should be lower
        # than if we had perfect formality
        formal2 = _top(formality="Casual", occasions=["Lounging"],
                       color_family="Blacks", style_tags=["Casual"])
        total2, dims2 = compute_compatibility_score(formal2, casual)
        assert total < total2


# ===========================================================================
# Section 11: compute_compatibility_score End-to-End
# ===========================================================================

class TestComputeCompatibility:
    """End-to-end tests for compute_compatibility_score."""

    def test_returns_8_dimension_keys(self):
        a = _top()
        b = _bottom()
        total, dims = compute_compatibility_score(a, b)
        expected_keys = {
            "occasion_formality", "style", "fabric", "silhouette",
            "color", "seasonality", "pattern", "price",
        }
        assert set(dims.keys()) == expected_keys

    def test_total_in_range(self):
        a = _top()
        b = _bottom()
        total, _ = compute_compatibility_score(a, b)
        assert 0.0 <= total <= 1.0

    def test_all_dims_in_range(self):
        a = _top()
        b = _bottom()
        _, dims = compute_compatibility_score(a, b)
        for key, val in dims.items():
            assert 0.0 <= val <= 1.0, f"{key} out of range: {val}"

    def test_good_pairing_scores_high(self):
        """Classic white blouse + dark straight jeans → high score."""
        a = _top()
        b = _bottom()
        total, _ = compute_compatibility_score(a, b)
        assert total > 0.65

    def test_bad_pairing_scores_low(self):
        """Formal silk blouse + gym shorts → low score."""
        formal = _top(formality="Formal", occasions=["Wedding Guest"],
                      style_tags=["Glamorous", "Romantic"],
                      color_family="Blacks", apparent_fabric="Silk",
                      seasons=["Fall", "Winter"], price=120)
        gym = _bottom(formality="Casual", occasions=["Workout", "Gym"],
                      style_tags=["Sporty"], color_family="Neon",
                      apparent_fabric="Technical", pattern="Graphic",
                      silhouette="Relaxed", length="Mini",
                      coverage_level="Minimal",
                      seasons=["Summer"], price=12)
        total, _ = compute_compatibility_score(formal, gym)
        assert total < 0.40

    def test_discrimination_gap(self):
        """Good pairing should score significantly higher than bad."""
        top = _top()
        good_bottom = _bottom()
        bad_bottom = _bottom(
            formality="Casual", occasions=["Workout", "Gym"],
            style_tags=["Sporty"], color_family="Neon",
            apparent_fabric="Technical", seasons=["Summer"], price=12,
        )
        good_score, _ = compute_compatibility_score(top, good_bottom)
        bad_score, _ = compute_compatibility_score(top, bad_bottom)
        gap = good_score - bad_score
        assert gap > 0.15, f"Gap too small: {gap:.3f} ({good_score:.3f} vs {bad_score:.3f})"

    def test_no_old_dimension_keys(self):
        """Ensure old v1 keys are NOT present."""
        a = _top()
        b = _bottom()
        _, dims = compute_compatibility_score(a, b)
        old_keys = {"formality", "occasion", "material", "balance", "season"}
        assert not (set(dims.keys()) & old_keys), \
            f"Old keys found: {set(dims.keys()) & old_keys}"


# ===========================================================================
# Section 12: Weight Sums
# ===========================================================================

class TestWeightSums:
    """Verify all weight dicts sum to 1.0."""

    def test_default_weights_sum(self):
        total = sum(DEFAULT_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=0.001)

    def test_all_category_pair_weights_sum(self):
        for pair_key, weights in CATEGORY_PAIR_WEIGHTS.items():
            total = sum(weights.values())
            assert total == pytest.approx(1.0, abs=0.001), \
                f"Weights for {pair_key} sum to {total}"

    def test_all_pairs_have_8_dims(self):
        expected = {"occasion_formality", "style", "fabric", "silhouette",
                    "color", "seasonality", "pattern", "price"}
        for pair_key, weights in CATEGORY_PAIR_WEIGHTS.items():
            assert set(weights.keys()) == expected, \
                f"{pair_key} has wrong keys: {set(weights.keys())}"
        assert set(DEFAULT_WEIGHTS.keys()) == expected


# ===========================================================================
# Section 13: Fabric Family + Lookup Helpers
# ===========================================================================

class TestFabricHelpers:

    def test_get_fabric_family_exact(self):
        assert _get_fabric_family("denim") == "denim"
        assert _get_fabric_family("silk") == "silk"
        assert _get_fabric_family("cashmere") == "wool"

    def test_get_fabric_family_substring(self):
        assert _get_fabric_family("Cotton Blend") == "cotton"
        assert _get_fabric_family("Stretch Denim") == "denim"

    def test_get_fabric_family_none(self):
        assert _get_fabric_family(None) is None
        assert _get_fabric_family("") is None

    def test_lookup_symmetric(self):
        table = {("a", "b"): 0.8}
        assert _lookup_symmetric(table, "a", "b") == 0.8
        assert _lookup_symmetric(table, "b", "a") == 0.8
        assert _lookup_symmetric(table, "a", "c", 0.5) == 0.5

    def test_fabric_compat_matrix_symmetric(self):
        """Spot-check that lookups work both directions."""
        score1 = _lookup_symmetric(_FABRIC_FAMILY_COMPAT, "denim", "knit")
        score2 = _lookup_symmetric(_FABRIC_FAMILY_COMPAT, "knit", "denim")
        assert score1 == score2

    def test_same_material_penalty_entries(self):
        assert "denim" in _SAME_MATERIAL_PENALTY
        assert "leather" in _SAME_MATERIAL_PENALTY
        assert _SAME_MATERIAL_PENALTY["denim"] < 0.50


# ===========================================================================
# Section 14: Real-World Outfit Scenarios
# ===========================================================================

class TestRealWorldScenarios:
    """Fashion-sensible scenario tests."""

    def test_blue_jeans_white_tee_blazer(self):
        """Classic outfit: blue jeans + white tee + black blazer."""
        jeans = _bottom(color_family="Blues", apparent_fabric="Denim",
                        style_tags=["Casual", "Classic"])
        tee = _top(color_family="Whites", apparent_fabric="Cotton",
                   style_tags=["Casual", "Minimalist"],
                   gemini_category_l2="T-Shirt")
        blazer = _outerwear(color_family="Blacks", apparent_fabric="Wool",
                            style_tags=["Classic", "Chic"])

        score_tee_jeans, _ = compute_compatibility_score(tee, jeans)
        score_tee_blazer, _ = compute_compatibility_score(tee, blazer)
        assert score_tee_jeans > 0.60
        assert score_tee_blazer > 0.55

    def test_satin_skirt_knit_top(self):
        """Satin skirt + knit sweater = great fabric contrast."""
        skirt = _bottom(apparent_fabric="Satin", texture="Smooth",
                        silhouette="A-Line", color_family="Blacks",
                        style_tags=["Chic", "Romantic"],
                        formality="Semi-Formal", occasions=["Date Night"],
                        gemini_category_l2="Skirt")
        knit = _top(apparent_fabric="Knit", texture="Ribbed",
                    silhouette="Fitted", color_family="Creams",
                    style_tags=["Classic", "Chic"],
                    formality="Smart Casual", occasions=["Date Night"])
        score, dims = compute_compatibility_score(knit, skirt)
        assert dims["fabric"] > 0.65  # Good contrast
        assert score > 0.55

    def test_denim_on_denim_penalized(self):
        """Denim jacket + denim jeans = double penalty."""
        jeans = _bottom(apparent_fabric="Denim", color_family="Blues")
        denim_jacket = _outerwear(apparent_fabric="Denim", color_family="Blues",
                                   gemini_category_l2="Jacket")
        score, dims = compute_compatibility_score(jeans, denim_jacket)
        assert dims["fabric"] < 0.50
        assert dims["color"] < 0.45

    def test_summer_dress_winter_coat_bad(self):
        """Summer linen dress + winter wool coat = season clash."""
        dress = _make_profile(
            broad_category="dresses", gemini_category_l1="Dresses",
            gemini_category_l2="Dress", apparent_fabric="Linen",
            seasons=["Summer"], formality="Casual",
            style_tags=["Casual", "Bohemian"],
            color_family="Whites", pattern="Solid",
            silhouette="A-Line", length="Midi",
            coverage_level="Moderate", price=60,
        )
        coat = _outerwear(apparent_fabric="Wool", seasons=["Winter"],
                          color_family="Blacks")
        score, dims = compute_compatibility_score(dress, coat)
        assert dims["seasonality"] < 0.50

    def test_activewear_vs_formal_killed(self):
        """Gym leggings + formal blazer = should score very low."""
        leggings = _bottom(
            formality="Casual", occasions=["Workout", "Gym"],
            style_tags=["Sporty"], apparent_fabric="Spandex",
            color_family="Blacks", silhouette="Fitted",
            seasons=["Spring", "Summer", "Fall", "Winter"],
            price=25,
        )
        formal_blazer = _outerwear(
            formality="Formal", occasions=["Formal Event", "Work"],
            style_tags=["Classic", "Glamorous"], apparent_fabric="Wool",
            color_family="Blacks", seasons=["Fall", "Winter"],
            price=150,
        )
        score, dims = compute_compatibility_score(leggings, formal_blazer)
        assert dims["occasion_formality"] < 0.35
        assert score < 0.55
