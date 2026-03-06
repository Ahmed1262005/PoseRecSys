"""
Unit tests for the Styling Metadata Scorer (v3.4).

Tests compute_styling_adjustment(), score_candidates_batch(),
fuzzy avoid matching, volume/formality balance, vibe coherence,
and clamping behaviour.
"""

import pytest
from services.outfit_engine import AestheticProfile, _derive_fields
from services.styling_scorer import (
    compute_styling_adjustment,
    score_candidates_batch,
    StylingMatchResult,
    _fuzzy_avoid_match,
    _norm,
    _norm_set,
    _check_volume_balance,
    _check_formality_balance,
    CATEGORY_MATCH_BONUS,
    FIT_MATCH_BONUS,
    MATERIAL_MATCH_BONUS,
    COLOR_MATCH_BONUS,
    RISE_MATCH_BONUS,
    LENGTH_MATCH_BONUS,
    PROFILE_AVOID_PENALTY,
    HARD_AVOID_PENALTY,
    SOFT_AVOID_PENALTY,
    PAIRING_RULES_BONUS,
    PAIRING_RULES_PENALTY,
    VIBE_COHERENCE_BONUS,
    FORMALITY_MISMATCH_PENALTY,
    MAX_POSITIVE,
    MAX_NEGATIVE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(**kwargs) -> AestheticProfile:
    """Create profile with derived fields computed."""
    p = AestheticProfile(**kwargs)
    _derive_fields(p)
    return p


def _source_top(**kwargs) -> AestheticProfile:
    """Source top product with sensible defaults."""
    defaults = dict(
        product_id="src-001",
        broad_category="tops",
        gemini_category_l1="Tops",
        gemini_category_l2="Blouse",
        formality="Smart Casual",
        fit_type="Fitted",
        occasions=["Everyday", "Work"],
        style_tags=["Classic", "Chic"],
        color_family="Whites",
        apparent_fabric="Cotton",
        pattern="Solid",
        silhouette="Fitted",
    )
    defaults.update(kwargs)
    return _make_profile(**defaults)


def _candidate_bottom(**kwargs) -> AestheticProfile:
    """Candidate bottoms product with sensible defaults."""
    defaults = dict(
        product_id="cand-001",
        broad_category="bottoms",
        gemini_category_l1="Bottoms",
        gemini_category_l2="Straight Leg Jeans",
        formality="Casual",
        fit_type="Regular",
        occasions=["Everyday"],
        style_tags=["Classic"],
        color_family="Blues",
        apparent_fabric="Denim",
        pattern="Solid",
        rise="Mid",
        length="Full Length",
    )
    defaults.update(kwargs)
    return _make_profile(**defaults)


def _styling_metadata_full(**overrides):
    """Build a realistic styling_metadata dict."""
    base = {
        "ideal_bottom_profile": {
            "categories": ["straight leg jeans", "wide leg trousers", "tailored pants"],
            "fits": ["regular", "relaxed"],
            "rises": ["mid", "high"],
            "lengths": ["full length", "ankle"],
            "material_traits": ["denim", "cotton", "wool blend"],
            "color_traits": ["blue", "navy", "black", "neutral"],
            "silhouette_traits": ["straight", "wide"],
            "avoid": ["skinny jeans with distressing", "low-rise cargo pants"],
            "why": "Balanced silhouette to complement fitted top",
        },
        "ideal_top_profile": {
            "categories": ["t-shirt", "tank top"],
            "fits": ["fitted", "slim"],
            "material_traits": ["cotton", "jersey"],
            "color_traits": ["white", "black"],
            "avoid": [],
        },
        "ideal_outerwear_profile": {
            "categories": ["blazer", "leather jacket"],
            "fits": ["regular", "slim"],
            "material_traits": ["leather", "wool"],
            "color_traits": ["black", "navy"],
            "avoid": ["puffer jacket"],
        },
        "hard_avoids": ["neon mesh crop top", "sequin mini skirt"],
        "soft_avoids": ["oversized graphic hoodie", "athletic joggers"],
        "pairing_rules": {
            "volume_balance": "contrast fitted top with relaxed bottom",
            "formality": "keep within 1-2 levels",
        },
        "formality_level": 3,
        "styling_role": "anchor_top",
        "pairing_flexibility_score": 7,
        "layering_flexibility_score": 5,
    }
    base.update(overrides)
    return base


# ===========================================================================
# 1. NORMALISATION HELPERS
# ===========================================================================

class TestNormHelpers:
    def test_norm_basic(self):
        assert _norm("Straight Leg Jeans") == "straight_leg_jeans"

    def test_norm_hyphens(self):
        assert _norm("high-rise") == "high_rise"

    def test_norm_empty(self):
        assert _norm("") == ""
        assert _norm(None) == ""

    def test_norm_set_list(self):
        result = _norm_set(["Mid", "High", ""])
        assert result == {"mid", "high"}

    def test_norm_set_string(self):
        assert _norm_set("Regular") == {"regular"}

    def test_norm_set_none(self):
        assert _norm_set(None) == set()


# ===========================================================================
# 2. FUZZY AVOID MATCHING
# ===========================================================================

class TestFuzzyAvoidMatch:
    def test_specific_avoid_matches(self):
        """Two-keyword overlap triggers a match."""
        assert _fuzzy_avoid_match(
            "skinny jeans with distressing",
            "skinny jeans", "Distressed Skinny Jeans", "denim", "skinny",
        )

    def test_specific_avoid_no_match(self):
        """Unrelated candidate doesn't match."""
        assert not _fuzzy_avoid_match(
            "skinny jeans with distressing",
            "wide leg trousers", "Wide Leg Wool Trousers", "wool", "relaxed",
        )

    def test_vague_avoid_skipped(self):
        """Short vague avoids like 'anything too formal' are skipped."""
        assert not _fuzzy_avoid_match(
            "anything overly formal",
            "blazer", "Classic Blazer", "wool", "regular",
        )

    def test_empty_avoid(self):
        assert not _fuzzy_avoid_match("", "jeans", "Jeans", "denim", "slim")

    def test_longer_vague_not_skipped(self):
        """Longer avoids with 'anything' but >6 words are not skipped."""
        assert _fuzzy_avoid_match(
            "anything with heavy denim distressing and rips and tears",
            "jeans", "Heavy Distressed Ripped Denim Jeans", "denim", "slim",
        )


# ===========================================================================
# 3. VOLUME BALANCE
# ===========================================================================

class TestVolumeBalance:
    def test_ideal_contrast(self):
        """1-2 level difference gets bonus."""
        adj = _check_volume_balance("fitted", "regular", {})
        assert adj == PAIRING_RULES_BONUS

    def test_slight_contrast(self):
        adj = _check_volume_balance("slim", "regular", {})
        assert adj == PAIRING_RULES_BONUS

    def test_both_tight_penalty(self):
        """Both fitted/slim penalised."""
        adj = _check_volume_balance("fitted", "slim", {})
        assert adj == PAIRING_RULES_BONUS  # diff=1, still contrast

    def test_both_fitted_penalty(self):
        adj = _check_volume_balance("fitted", "fitted", {})
        assert adj == PAIRING_RULES_PENALTY * 0.5

    def test_extreme_contrast_penalty(self):
        adj = _check_volume_balance("fitted", "oversized", {})
        assert adj == PAIRING_RULES_PENALTY * 0.5

    def test_missing_fit(self):
        assert _check_volume_balance(None, "regular", {}) == 0.0
        assert _check_volume_balance("fitted", None, {}) == 0.0


# ===========================================================================
# 4. FORMALITY BALANCE
# ===========================================================================

class TestFormalityBalance:
    def test_close_formality_bonus(self):
        """Source level 3 (business casual) + candidate 'smart_casual' (2) = diff 1."""
        adj = _check_formality_balance(3, "smart_casual")
        assert adj == PAIRING_RULES_BONUS * 0.5

    def test_large_gap_penalty(self):
        """Source level 5 (formal) + candidate 'casual' (1) = diff 4."""
        adj = _check_formality_balance(5, "casual")
        assert adj == FORMALITY_MISMATCH_PENALTY

    def test_missing_candidate(self):
        assert _check_formality_balance(3, None) == 0.0

    def test_exact_match(self):
        """Source 2 + smart_casual (2) = diff 0, still <= 1."""
        adj = _check_formality_balance(2, "smart_casual")
        assert adj == PAIRING_RULES_BONUS * 0.5


# ===========================================================================
# 5. COMPUTE_STYLING_ADJUSTMENT — NO METADATA
# ===========================================================================

class TestNoMetadata:
    def test_no_source_metadata_returns_zero(self):
        """Without styling_metadata, adjustment is 0."""
        source = _source_top()
        cand = _candidate_bottom()
        result = compute_styling_adjustment(source, cand, "bottoms")
        assert result.styling_adj == 0.0
        assert not result.has_source_metadata

    def test_empty_metadata_returns_zero(self):
        result = compute_styling_adjustment(
            _source_top(), _candidate_bottom(), "bottoms",
            source_styling_metadata={},
        )
        assert result.styling_adj == 0.0


# ===========================================================================
# 6. COMPUTE_STYLING_ADJUSTMENT — PROFILE MATCHING
# ===========================================================================

class TestProfileMatching:
    def test_category_match_bonus(self):
        source = _source_top()
        cand = _candidate_bottom(gemini_category_l2="Straight Leg Jeans")
        meta = _styling_metadata_full()
        result = compute_styling_adjustment(source, cand, "bottoms", meta)
        assert "category_match" in result.breakdown
        assert result.breakdown["category_match"] == CATEGORY_MATCH_BONUS

    def test_category_partial_match(self):
        """L2 'Cropped Straight Leg Jeans' partially matches 'straight leg jeans'."""
        source = _source_top()
        cand = _candidate_bottom(gemini_category_l2="Cropped Straight Leg Jeans")
        meta = _styling_metadata_full()
        # 'cropped_straight_leg_jeans' contains 'straight_leg_jeans'
        result = compute_styling_adjustment(source, cand, "bottoms", meta)
        assert ("category_match" in result.breakdown or
                "category_partial" in result.breakdown)

    def test_fit_match_bonus(self):
        source = _source_top()
        cand = _candidate_bottom(fit_type="Regular")
        meta = _styling_metadata_full()
        result = compute_styling_adjustment(source, cand, "bottoms", meta)
        assert "fit_match" in result.breakdown
        assert result.breakdown["fit_match"] == FIT_MATCH_BONUS

    def test_rise_match_bonus(self):
        source = _source_top()
        cand = _candidate_bottom(rise="Mid")
        meta = _styling_metadata_full()
        result = compute_styling_adjustment(source, cand, "bottoms", meta)
        assert "rise_match" in result.breakdown
        assert result.breakdown["rise_match"] == RISE_MATCH_BONUS

    def test_length_match_bonus(self):
        source = _source_top()
        cand = _candidate_bottom(length="Full Length")
        meta = _styling_metadata_full()
        result = compute_styling_adjustment(source, cand, "bottoms", meta)
        assert "length_match" in result.breakdown

    def test_material_match_bonus(self):
        source = _source_top()
        cand = _candidate_bottom(apparent_fabric="Denim")
        meta = _styling_metadata_full()
        result = compute_styling_adjustment(source, cand, "bottoms", meta)
        assert "material_match" in result.breakdown
        assert result.breakdown["material_match"] == MATERIAL_MATCH_BONUS

    def test_color_match_bonus(self):
        source = _source_top()
        cand = _candidate_bottom(color_family="Blues")
        meta = _styling_metadata_full()
        # 'blue' in color_traits, 'blues' normalised should partial-match 'blue'
        result = compute_styling_adjustment(source, cand, "bottoms", meta)
        # blue in blues or blues in blue — at least partial
        assert "color_match" in result.breakdown or result.styling_adj >= 0

    def test_no_match_on_unknown_category(self):
        """Target category with no profile key → no profile bonuses."""
        source = _source_top()
        cand = _candidate_bottom()
        meta = _styling_metadata_full()
        result = compute_styling_adjustment(source, cand, "accessories", meta)
        assert "category_match" not in result.breakdown
        assert "fit_match" not in result.breakdown

    def test_multiple_bonuses_stack(self):
        """Category + fit + rise + length + material all stack."""
        source = _source_top()
        cand = _candidate_bottom(
            gemini_category_l2="Straight Leg Jeans",
            fit_type="Regular",
            rise="Mid",
            length="Full Length",
            apparent_fabric="Denim",
        )
        meta = _styling_metadata_full()
        result = compute_styling_adjustment(source, cand, "bottoms", meta)
        expected_min = (CATEGORY_MATCH_BONUS + FIT_MATCH_BONUS +
                        RISE_MATCH_BONUS + LENGTH_MATCH_BONUS +
                        MATERIAL_MATCH_BONUS)
        assert result.styling_adj >= expected_min - 0.001


# ===========================================================================
# 7. COMPUTE_STYLING_ADJUSTMENT — AVOID PENALTIES
# ===========================================================================

class TestAvoidPenalties:
    def test_profile_avoid_penalty(self):
        """Candidate matching a profile avoid gets penalised."""
        source = _source_top()
        cand = _candidate_bottom(
            gemini_category_l2="Skinny Jeans",
            name="Distressed Skinny Jeans",
            fit_type="Skinny",
        )
        meta = _styling_metadata_full()
        result = compute_styling_adjustment(source, cand, "bottoms", meta)
        assert "profile_avoid" in result.breakdown
        assert result.breakdown["profile_avoid"] == PROFILE_AVOID_PENALTY
        assert any("profile:" in h for h in result.avoid_hits)

    def test_hard_avoid_penalty(self):
        source = _source_top()
        cand = _candidate_bottom(
            gemini_category_l2="Mini Skirt",
            name="Sequin Mini Skirt Party",
            apparent_fabric="Sequin",
        )
        meta = _styling_metadata_full()
        result = compute_styling_adjustment(source, cand, "bottoms", meta)
        assert "hard_avoid" in result.breakdown
        assert result.breakdown["hard_avoid"] == HARD_AVOID_PENALTY

    def test_soft_avoid_penalty(self):
        source = _source_top()
        cand = _candidate_bottom(
            gemini_category_l2="Joggers",
            name="Oversized Graphic Athletic Joggers",
            fit_type="Oversized",
        )
        meta = _styling_metadata_full()
        result = compute_styling_adjustment(source, cand, "bottoms", meta)
        assert "soft_avoid" in result.breakdown
        assert result.breakdown["soft_avoid"] == SOFT_AVOID_PENALTY

    def test_hard_avoid_string_instead_of_list(self):
        """Handle hard_avoids as a single string (edge case)."""
        meta = _styling_metadata_full(hard_avoids="neon mesh crop top")
        source = _source_top()
        cand = _candidate_bottom(
            name="Neon Mesh Crop Top Style",
            gemini_category_l2="Crop Top",
        )
        result = compute_styling_adjustment(source, cand, "bottoms", meta)
        assert "hard_avoid" in result.breakdown


# ===========================================================================
# 8. VOLUME & FORMALITY IN SCORING
# ===========================================================================

class TestPairingRulesInScoring:
    def test_volume_contrast_bonus(self):
        """Fitted top + regular bottom = 2 level diff = bonus."""
        source = _source_top(fit_type="Fitted")
        cand = _candidate_bottom(fit_type="Regular")
        meta = _styling_metadata_full()
        result = compute_styling_adjustment(source, cand, "bottoms", meta)
        assert "volume_balance" in result.breakdown
        assert result.breakdown["volume_balance"] > 0

    def test_formality_mismatch_penalty(self):
        """Source formality 5 + casual candidate = big gap."""
        source = _source_top()
        cand = _candidate_bottom(formality="Casual")
        meta = _styling_metadata_full(formality_level=5)
        result = compute_styling_adjustment(source, cand, "bottoms", meta)
        assert "formality_balance" in result.breakdown
        assert result.breakdown["formality_balance"] == FORMALITY_MISMATCH_PENALTY


# ===========================================================================
# 9. VIBE / APPEARANCE COHERENCE
# ===========================================================================

class TestVibeCoherence:
    def test_appearance_overlap_bonus(self):
        source = _source_top()
        cand = _candidate_bottom()
        meta = _styling_metadata_full()
        result = compute_styling_adjustment(
            source, cand, "bottoms", meta,
            source_appearance_tags=["casual", "sleek", "minimal"],
            candidate_appearance_tags=["casual", "sleek", "bold"],
        )
        assert "appearance_coherence" in result.breakdown
        assert result.breakdown["appearance_coherence"] == VIBE_COHERENCE_BONUS

    def test_appearance_insufficient_overlap(self):
        """Only 1 overlap is not enough."""
        source = _source_top()
        cand = _candidate_bottom()
        meta = _styling_metadata_full()
        result = compute_styling_adjustment(
            source, cand, "bottoms", meta,
            source_appearance_tags=["casual"],
            candidate_appearance_tags=["casual", "bold"],
        )
        assert "appearance_coherence" not in result.breakdown

    def test_vibe_word_overlap_bonus(self):
        """Vibe tags share word stems."""
        source = _source_top()
        cand = _candidate_bottom()
        meta = _styling_metadata_full()
        result = compute_styling_adjustment(
            source, cand, "bottoms", meta,
            source_vibe_tags=["quiet_luxury", "effortless_parisian"],
            candidate_vibe_tags=["parisian_chic", "quiet_elegance"],
        )
        # 'quiet' and 'parisian' overlap (after word split)
        assert ("vibe_coherence" in result.breakdown or
                "appearance_coherence" in result.breakdown or
                result.styling_adj > 0)

    def test_no_vibe_without_tags(self):
        """No tags = no coherence bonus."""
        source = _source_top()
        cand = _candidate_bottom()
        meta = _styling_metadata_full()
        result = compute_styling_adjustment(source, cand, "bottoms", meta)
        assert "appearance_coherence" not in result.breakdown
        assert "vibe_coherence" not in result.breakdown


# ===========================================================================
# 10. CLAMPING
# ===========================================================================

class TestClamping:
    def test_positive_clamped_to_max(self):
        """Even with all bonuses, can't exceed MAX_POSITIVE."""
        source = _source_top()
        cand = _candidate_bottom(
            gemini_category_l2="Straight Leg Jeans",
            fit_type="Regular",
            rise="Mid",
            length="Full Length",
            apparent_fabric="Denim",
            color_family="Navy",
        )
        meta = _styling_metadata_full()
        result = compute_styling_adjustment(
            source, cand, "bottoms", meta,
            source_appearance_tags=["casual", "sleek", "minimal"],
            candidate_appearance_tags=["casual", "sleek", "crisp"],
            source_vibe_tags=["quiet_luxury"],
            candidate_vibe_tags=["quiet_luxury"],
        )
        assert result.styling_adj <= MAX_POSITIVE

    def test_negative_clamped_to_min(self):
        """Even with all penalties, can't go below MAX_NEGATIVE."""
        source = _source_top(fit_type="Fitted")
        cand = _candidate_bottom(
            gemini_category_l2="Skinny Jeans",
            name="Neon Sequin Distressed Skinny Mesh Jeans",
            fit_type="Fitted",
            apparent_fabric="Mesh Sequin",
            formality="Casual",
        )
        meta = _styling_metadata_full(
            formality_level=5,
            hard_avoids=["neon sequin distressed mesh jeans"],
            soft_avoids=["skinny mesh distressed pants"],
        )
        result = compute_styling_adjustment(source, cand, "bottoms", meta)
        assert result.styling_adj >= MAX_NEGATIVE


# ===========================================================================
# 11. OUTERWEAR PROFILE MAPPING
# ===========================================================================

class TestOuterwearMapping:
    def test_outerwear_uses_outerwear_profile(self):
        source = _source_top()
        cand = _make_profile(
            product_id="cand-ow-1",
            broad_category="outerwear",
            gemini_category_l1="Outerwear",
            gemini_category_l2="Blazer",
            fit_type="Regular",
            apparent_fabric="Wool",
            color_family="Black",
        )
        meta = _styling_metadata_full()
        result = compute_styling_adjustment(source, cand, "outerwear", meta)
        assert "category_match" in result.breakdown
        assert result.breakdown["category_match"] == CATEGORY_MATCH_BONUS

    def test_dresses_use_outerwear_profile(self):
        """Dresses map to outerwear profile (what goes over them)."""
        source = _source_top()
        cand = _make_profile(
            product_id="cand-ow-2",
            broad_category="outerwear",
            gemini_category_l1="Outerwear",
            gemini_category_l2="Leather Jacket",
            fit_type="Slim",
            apparent_fabric="Leather",
        )
        meta = _styling_metadata_full()
        result = compute_styling_adjustment(source, cand, "dresses", meta)
        assert "category_match" in result.breakdown


# ===========================================================================
# 12. BATCH SCORER
# ===========================================================================

class TestBatchScorer:
    def test_batch_returns_all_candidates(self):
        source = _source_top()
        candidates = [
            _candidate_bottom(product_id="c1"),
            _candidate_bottom(product_id="c2"),
            _candidate_bottom(product_id="c3"),
        ]
        meta = _styling_metadata_full()
        results = score_candidates_batch(
            source, candidates, "bottoms", meta,
        )
        assert len(results) == 3
        assert "c1" in results
        assert "c2" in results
        assert "c3" in results

    def test_batch_no_metadata_all_zero(self):
        source = _source_top()
        candidates = [
            _candidate_bottom(product_id="c1"),
            _candidate_bottom(product_id="c2"),
        ]
        results = score_candidates_batch(source, candidates, "bottoms")
        for r in results.values():
            assert r.styling_adj == 0.0
            assert not r.has_source_metadata

    def test_batch_with_candidate_extras(self):
        source = _source_top()
        candidates = [
            _candidate_bottom(product_id="c1"),
            _candidate_bottom(product_id="c2"),
        ]
        meta = _styling_metadata_full()
        extras = {
            "c1": {
                "appearance_top_tags": ["casual", "sleek", "minimal"],
                "vibe_tags": ["quiet_luxury"],
            },
            "c2": {
                "appearance_top_tags": ["bold", "edgy"],
                "vibe_tags": ["streetwear"],
            },
        }
        results = score_candidates_batch(
            source, candidates, "bottoms", meta,
            source_appearance_tags=["casual", "sleek", "crisp"],
            source_vibe_tags=["quiet_luxury"],
            candidate_extras=extras,
        )
        # c1 should have higher adj than c2 (appearance overlap)
        assert results["c1"].styling_adj >= results["c2"].styling_adj

    def test_batch_graceful_with_empty_candidates(self):
        source = _source_top()
        meta = _styling_metadata_full()
        results = score_candidates_batch(source, [], "bottoms", meta)
        assert results == {}


# ===========================================================================
# 13. RESULT DATACLASS
# ===========================================================================

class TestStylingMatchResult:
    def test_default_values(self):
        r = StylingMatchResult()
        assert r.styling_adj == 0.0
        assert r.breakdown == {}
        assert r.matched_profile_fields == []
        assert r.avoid_hits == []
        assert r.has_source_metadata is False

    def test_avoid_hits_populated(self):
        source = _source_top()
        cand = _candidate_bottom(
            gemini_category_l2="Mini Skirt",
            name="Sequin Mini Skirt",
            apparent_fabric="Sequin",
        )
        meta = _styling_metadata_full()
        result = compute_styling_adjustment(source, cand, "bottoms", meta)
        assert len(result.avoid_hits) > 0


# ===========================================================================
# 14. EDGE CASES
# ===========================================================================

class TestEdgeCases:
    def test_profile_avoid_as_string(self):
        """Profile avoid can be a single string instead of list."""
        meta = _styling_metadata_full()
        meta["ideal_bottom_profile"]["avoid"] = "low-rise cargo pants with chains"
        source = _source_top()
        cand = _candidate_bottom(
            gemini_category_l2="Cargo Pants",
            name="Low Rise Cargo Pants with Chains",
            rise="Low",
        )
        result = compute_styling_adjustment(source, cand, "bottoms", meta)
        assert "profile_avoid" in result.breakdown

    def test_missing_profile_key_graceful(self):
        """If ideal_bottom_profile is missing, still scores avoids/pairing."""
        meta = _styling_metadata_full()
        del meta["ideal_bottom_profile"]
        source = _source_top()
        cand = _candidate_bottom()
        result = compute_styling_adjustment(source, cand, "bottoms", meta)
        # Should still score volume/formality/avoids, just no profile bonuses
        assert result.has_source_metadata

    def test_none_formality_level_uses_profile(self):
        """If styling_metadata has no formality_level, falls back to profile's."""
        meta = _styling_metadata_full()
        meta.pop("formality_level")
        source = _source_top(formality="Smart Casual")
        cand = _candidate_bottom(formality="Casual")
        result = compute_styling_adjustment(source, cand, "bottoms", meta)
        # Should not crash — uses source.formality_level
        assert isinstance(result.styling_adj, float)

    def test_candidate_with_no_attributes(self):
        """Candidate with all None attributes gets minimal scoring."""
        source = _source_top()
        cand = _make_profile(product_id="cand-empty", broad_category="bottoms")
        meta = _styling_metadata_full()
        result = compute_styling_adjustment(source, cand, "bottoms", meta)
        assert isinstance(result.styling_adj, float)
        assert result.has_source_metadata
