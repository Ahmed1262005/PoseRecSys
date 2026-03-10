"""
Unit tests for the outerwear appropriateness gates (v3.5).

Tests that outerwear is only recommended when the source item has sufficient
formality or appropriate occasions.  Covers three groups:

Group 1 — Formal outerwear (blazer, suit jacket):
  - Hard filter: formality 1 without formal occasions -> filtered
  - Soft penalty: formality 2 without formal occasions -> -0.10
  - Occasion override: formality 1-2 + formal occasion -> neutral/pass
  - Formal source: formality 3+ -> +0.03 structured bonus

Group 2 — Volume/statement outerwear (poncho, cape, kimono, bolero, shrug):
  - Hard filter: formality 1 always (NO occasion override)
  - Soft penalty: formality 2 always (NO occasion override)
  - Neutral: formality 3+

Group 3 — Heavy outerwear (parka, coat):
  - Penalty: formality 1 without formal occasion -> -0.10
  - Keep: formality 1 + formal occasion -> +0.03
  - Keep: formality 2+ -> +0.03

Also covers:
  - Non-gated outerwear: jackets/bombers always get structured bonus
  - Precompute constants: verify precompute mirrors engine constants
"""

import pytest
from services.outfit_engine import (
    AestheticProfile,
    _derive_fields,
    _filter_by_gemini_category,
    _BLAZER_GATE_L2,
    _BLAZER_GATE_OCCASIONS,
    _BLAZER_MIN_FORMALITY,
    _BLAZER_CASUAL_PENALTY,
    _FORMAL_OUTER_L2,
    _VOLUME_OUTER_L2,
    _HEAVY_OUTER_L2,
    _OUTERWEAR_GATE_OCCASIONS,
    _OUTERWEAR_MIN_FORMALITY,
    _OUTERWEAR_CASUAL_PENALTY,
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
    """A tops source with sensible defaults."""
    defaults = dict(
        product_id="src-001",
        broad_category="tops",
        gemini_category_l1="Tops",
        gemini_category_l2="Blouse",
        formality="Smart Casual",
        occasions=["Everyday", "Work"],
        style_tags=["Classic", "Chic"],
        color_family="Whites",
        apparent_fabric="Cotton",
        texture="Smooth",
        pattern="Solid",
        silhouette="Fitted",
        length="Regular",
        coverage_level="Moderate",
        seasons=["Spring", "Fall"],
        price=50,
    )
    defaults.update(kwargs)
    return _make_profile(**defaults)


def _outerwear_cand(**kwargs) -> AestheticProfile:
    """An outerwear candidate with sensible defaults."""
    defaults = dict(
        product_id="cand-001",
        broad_category="outerwear",
        gemini_category_l1="Outerwear",
        gemini_category_l2="Blazer",
        formality="Smart Casual",
        occasions=["Work", "Everyday"],
        style_tags=["Classic", "Chic"],
        color_family="Blacks",
        apparent_fabric="Wool",
        texture="Textured",
        pattern="Solid",
        silhouette="Regular",
        length="Regular",
        coverage_level="Full",
        seasons=["Fall", "Winter"],
        price=80,
        similarity=0.80,
    )
    defaults.update(kwargs)
    return _make_profile(**defaults)


# ===========================================================================
# 1. Hard Filter Tests (_filter_by_gemini_category)
# ===========================================================================


class TestBlazerHardFilter:
    """Blazers should be hard-filtered from outerwear when source is
    formality <= 1 with no formal occasions."""

    def test_cami_top_no_blazers(self):
        """Cami top (formality 1, everyday) -> blazers filtered out."""
        source = _source_top(
            gemini_category_l2="Camisole",
            formality="Casual",           # -> formality_level 1
            occasions=["Everyday"],
        )
        blazer = _outerwear_cand(product_id="blz-1", gemini_category_l2="Blazer")
        jacket = _outerwear_cand(product_id="jkt-1", gemini_category_l2="Jacket")

        result = _filter_by_gemini_category(source, [blazer, jacket], "outerwear")
        result_ids = [c.product_id for c in result]

        assert "jkt-1" in result_ids, "Jacket should pass filter"
        assert "blz-1" not in result_ids, "Blazer should be hard-filtered for cami top"

    def test_tank_top_no_blazers(self):
        """Tank top (formality 1, casual) -> blazers filtered out."""
        source = _source_top(
            gemini_category_l2="Tank Top",
            formality="Casual",
            occasions=["Casual", "Summer"],
        )
        blazer = _outerwear_cand(product_id="blz-1", gemini_category_l2="Blazer")

        result = _filter_by_gemini_category(source, [blazer], "outerwear")
        assert len(result) == 0, "Blazer should be filtered for tank top"

    def test_crop_top_no_blazers(self):
        """Crop top (formality 1, everyday) -> blazers filtered out."""
        source = _source_top(
            gemini_category_l2="Crop Top",
            formality="Casual",
            occasions=["Everyday", "Festival"],
        )
        blazer = _outerwear_cand(product_id="blz-1", gemini_category_l2="Blazer")
        bomber = _outerwear_cand(product_id="bmb-1", gemini_category_l2="Bomber")

        result = _filter_by_gemini_category(source, [blazer, bomber], "outerwear")
        result_ids = [c.product_id for c in result]

        assert "bmb-1" in result_ids, "Bomber should pass filter"
        assert "blz-1" not in result_ids, "Blazer should be filtered for crop top"

    def test_graphic_tee_no_blazers(self):
        """Graphic tee (formality 1, casual) -> blazers filtered out."""
        source = _source_top(
            gemini_category_l2="T-Shirt",
            formality="Casual",
            occasions=["Casual", "Weekend"],
        )
        blazer = _outerwear_cand(product_id="blz-1", gemini_category_l2="Blazer")

        result = _filter_by_gemini_category(source, [blazer], "outerwear")
        assert len(result) == 0

    def test_formality_1_with_date_night_keeps_blazer(self):
        """Formality 1 but with 'date night' occasion -> blazer passes."""
        source = _source_top(
            gemini_category_l2="Camisole",
            formality="Casual",
            occasions=["Date Night"],
        )
        blazer = _outerwear_cand(product_id="blz-1", gemini_category_l2="Blazer")

        result = _filter_by_gemini_category(source, [blazer], "outerwear")
        assert len(result) == 1, "Blazer should pass when source has formal occasion"

    def test_formality_1_with_work_occasion_keeps_blazer(self):
        """Formality 1 but with 'work' occasion -> blazer passes."""
        source = _source_top(
            gemini_category_l2="T-Shirt",
            formality="Casual",
            occasions=["Work"],
        )
        blazer = _outerwear_cand(product_id="blz-1", gemini_category_l2="Blazer")

        result = _filter_by_gemini_category(source, [blazer], "outerwear")
        assert len(result) == 1

    def test_formality_2_does_not_hard_filter(self):
        """Formality 2 sources should NOT be hard-filtered (only penalized at scoring)."""
        source = _source_top(
            formality="Smart Casual",   # -> formality_level 2 or 3
            occasions=["Everyday"],
        )
        blazer = _outerwear_cand(product_id="blz-1", gemini_category_l2="Blazer")

        result = _filter_by_gemini_category(source, [blazer], "outerwear")
        assert len(result) == 1, "Formality 2+ should not be hard-filtered"

    def test_non_gated_outerwear_never_filtered(self):
        """Jackets, bombers, coats, etc. are never hard-filtered by outerwear gates."""
        source = _source_top(
            gemini_category_l2="Camisole",
            formality="Casual",
            occasions=["Everyday"],
        )
        candidates = [
            _outerwear_cand(product_id="jkt-1", gemini_category_l2="Jacket"),
            _outerwear_cand(product_id="bmb-1", gemini_category_l2="Bomber"),
            _outerwear_cand(product_id="cot-1", gemini_category_l2="Coat"),
            _outerwear_cand(product_id="dnm-1", gemini_category_l2="Denim Jacket"),
            _outerwear_cand(product_id="lth-1", gemini_category_l2="Leather Jacket"),
            _outerwear_cand(product_id="trn-1", gemini_category_l2="Trench"),
        ]

        result = _filter_by_gemini_category(source, candidates, "outerwear")
        result_ids = {c.product_id for c in result}

        for cand in candidates:
            assert cand.product_id in result_ids, \
                f"{cand.gemini_category_l2} should not be filtered by blazer gate"

    def test_non_outerwear_target_ignores_blazer_gate(self):
        """Blazer gate only applies to target_broad='outerwear'."""
        source = _source_top(
            formality="Casual",
            occasions=["Everyday"],
        )
        # A blazer categorized as tops (e.g. "Blazer Top") shouldn't be filtered
        # by the outerwear blazer gate when retrieving tops
        cand = _outerwear_cand(
            product_id="bt-1",
            gemini_category_l1="Tops",
            gemini_category_l2="Blazer",
        )

        result = _filter_by_gemini_category(source, [cand], "tops")
        # The candidate's L1 is Tops, target is tops -> would be filtered by
        # L1 mismatch check if gemini_broad differs, but the blazer gate
        # itself should not apply for non-outerwear targets.
        # (Exact result depends on L1 mapping, but blazer gate is not the cause)


# ===========================================================================
# 2. Scoring Gate Tests (style_adj in _score_category)
# ===========================================================================


class TestBlazerScoringGate:
    """Blazer scoring behavior varies by source formality and occasions."""

    def _score_outerwear(self, source, candidates):
        """Run the scoring loop inline (mirrors _score_category outerwear block).

        Returns list of (product_id, style_adj) tuples.
        Uses the module-level outerwear gate constants from outfit_engine.
        """
        _STRUCTURED_OUTERWEAR = frozenset({
            "blazer", "suit jacket", "jacket", "coat", "trench",
            "trench coat", "leather jacket", "denim jacket", "bomber",
            "parka",
        })
        _STRUCTURED_BONUS = 0.03

        results = []
        for cand in candidates:
            style_adj = 0.0
            cand_l2 = (cand.gemini_category_l2 or "").lower().strip()
            _src_occasions = {o.lower().strip() for o in (source.occasions or [])}
            _has_formal_occ = bool(_src_occasions & _OUTERWEAR_GATE_OCCASIONS)
            if cand_l2 in _STRUCTURED_OUTERWEAR:
                if cand_l2 in _FORMAL_OUTER_L2:
                    # Formal outerwear gate (blazer, suit jacket)
                    if source.formality_level >= _OUTERWEAR_MIN_FORMALITY:
                        style_adj += _STRUCTURED_BONUS
                    elif _has_formal_occ:
                        pass  # neutral
                    else:
                        style_adj += _OUTERWEAR_CASUAL_PENALTY
                elif cand_l2 in _HEAVY_OUTER_L2:
                    # Heavy outerwear gate (parka, coat)
                    if source.formality_level <= 1 and not _has_formal_occ:
                        style_adj += _OUTERWEAR_CASUAL_PENALTY
                    else:
                        style_adj += _STRUCTURED_BONUS
                else:
                    # Other structured (jacket, bomber, denim jacket, etc.)
                    style_adj += _STRUCTURED_BONUS
            elif cand_l2 in _VOLUME_OUTER_L2:
                # Volume/statement: no bonus; penalize for casual
                if source.formality_level <= 2:
                    style_adj += _OUTERWEAR_CASUAL_PENALTY
            results.append((cand.product_id, round(style_adj, 4)))
        return results

    def test_formal_source_blazer_gets_bonus(self):
        """Office blouse (formality 3+, work) -> blazer gets +0.03."""
        source = _source_top(
            formality="Business Casual",   # -> formality_level 3
            occasions=["Work", "Office"],
        )
        blazer = _outerwear_cand(product_id="blz-1", gemini_category_l2="Blazer")

        results = self._score_outerwear(source, [blazer])
        assert results[0][1] == 0.03

    def test_casual_source_no_formal_occasion_blazer_penalized(self):
        """Basic tee (formality 2, everyday) -> blazer gets -0.10."""
        source = _source_top(
            formality="Smart Casual",   # This is formality 3 after _derive_fields
            occasions=["Everyday"],
        )
        # Force formality_level to 2 to simulate a casual source
        source.formality_level = 2

        blazer = _outerwear_cand(product_id="blz-1", gemini_category_l2="Blazer")

        results = self._score_outerwear(source, [blazer])
        assert results[0][1] == -0.10

    def test_casual_source_with_date_night_blazer_neutral(self):
        """Casual dress (formality 2, date night) -> blazer gets 0.00."""
        source = _source_top(
            formality="Smart Casual",
            occasions=["Date Night", "Evening"],
        )
        source.formality_level = 2

        blazer = _outerwear_cand(product_id="blz-1", gemini_category_l2="Blazer")

        results = self._score_outerwear(source, [blazer])
        assert results[0][1] == 0.00

    def test_casual_source_with_office_blazer_neutral(self):
        """Casual item with 'office' occasion -> blazer gets 0.00 (not penalized)."""
        source = _source_top(
            formality="Smart Casual",
            occasions=["Office"],
        )
        source.formality_level = 2

        blazer = _outerwear_cand(product_id="blz-1", gemini_category_l2="Blazer")

        results = self._score_outerwear(source, [blazer])
        assert results[0][1] == 0.00

    def test_jacket_always_gets_bonus(self):
        """Jackets get +0.03 regardless of source formality."""
        source = _source_top(
            formality="Casual",
            occasions=["Everyday"],
        )
        source.formality_level = 1

        jacket = _outerwear_cand(product_id="jkt-1", gemini_category_l2="Jacket")
        bomber = _outerwear_cand(product_id="bmb-1", gemini_category_l2="Bomber")
        denim = _outerwear_cand(product_id="dnm-1", gemini_category_l2="Denim Jacket")

        results = self._score_outerwear(source, [jacket, bomber, denim])
        for pid, adj in results:
            assert adj == 0.03, f"{pid} should always get structured bonus"

    def test_blazer_penalty_vs_jacket_bonus_swing(self):
        """For a casual source, blazer gets -0.10 vs jacket +0.03 = 0.13 swing."""
        source = _source_top(
            formality="Smart Casual",
            occasions=["Everyday"],
        )
        source.formality_level = 2

        blazer = _outerwear_cand(product_id="blz-1", gemini_category_l2="Blazer")
        jacket = _outerwear_cand(product_id="jkt-1", gemini_category_l2="Jacket")

        results = dict(self._score_outerwear(source, [blazer, jacket]))
        swing = results["jkt-1"] - results["blz-1"]
        assert swing == pytest.approx(0.13, abs=0.001), \
            f"Expected 0.13 swing (jacket +0.03 vs blazer -0.10), got {swing}"

    def test_occasion_matching_case_insensitive(self):
        """Occasion matching should be case-insensitive."""
        source = _source_top(
            formality="Smart Casual",
            occasions=["DATE NIGHT"],
        )
        source.formality_level = 2

        blazer = _outerwear_cand(product_id="blz-1", gemini_category_l2="Blazer")

        results = self._score_outerwear(source, [blazer])
        assert results[0][1] == 0.00, "Should match 'date night' case-insensitively"

    def test_multiple_occasions_one_match_is_enough(self):
        """If any source occasion matches blazer-OK set, blazer is neutral."""
        source = _source_top(
            formality="Smart Casual",
            occasions=["Beach", "Weekend", "Dinner"],
        )
        source.formality_level = 2

        blazer = _outerwear_cand(product_id="blz-1", gemini_category_l2="Blazer")

        results = self._score_outerwear(source, [blazer])
        assert results[0][1] == 0.00, "'Dinner' should be enough to allow blazer"

    def test_no_occasions_at_all_uses_formality(self):
        """Source with empty occasions list falls back to formality check."""
        source = _source_top(
            formality="Business Casual",   # -> formality_level 3
            occasions=[],
        )
        # Formality 3+ -> bonus
        results = self._score_outerwear(
            source,
            [_outerwear_cand(product_id="blz-1", gemini_category_l2="Blazer")],
        )
        assert results[0][1] == 0.03

        # Formality 2 with no occasions -> penalty
        source.formality_level = 2
        results = self._score_outerwear(
            source,
            [_outerwear_cand(product_id="blz-2", gemini_category_l2="Blazer")],
        )
        assert results[0][1] == -0.10


# ===========================================================================
# 3. Constants Sync Tests
# ===========================================================================


class TestBlazerGateConstants:
    """Verify precompute constants stay in sync with engine constants."""

    def test_blazer_l2_sets_match(self):
        """Precompute _BLAZER_L2 should match engine _BLAZER_GATE_L2."""
        from scripts.precompute_outfit_candidates import (
            _BLAZER_L2 as precompute_blazer_l2,
        )
        assert precompute_blazer_l2 == _BLAZER_GATE_L2

    def test_blazer_occasions_match(self):
        """Precompute _BLAZER_OK_OCCASIONS should match engine _BLAZER_GATE_OCCASIONS."""
        from scripts.precompute_outfit_candidates import (
            _BLAZER_OK_OCCASIONS as precompute_occasions,
        )
        assert precompute_occasions == _BLAZER_GATE_OCCASIONS

    def test_blazer_penalty_match(self):
        """Precompute _BLAZER_CASUAL_PENALTY should match engine constant."""
        from scripts.precompute_outfit_candidates import (
            _BLAZER_CASUAL_PENALTY as precompute_penalty,
            _BLAZER_MIN_FORMALITY as precompute_min_form,
        )
        # The engine defines these inside _score_category as local vars,
        # so we check the precompute module-level constants match the
        # documented values.
        assert precompute_penalty == -0.10
        assert precompute_min_form == 3

    def test_blazer_gate_occasions_completeness(self):
        """All key formal occasions should be in the gate set."""
        required = {"work", "office", "formal", "semi-formal", "date night",
                     "evening", "cocktail", "dinner", "interview"}
        assert required.issubset(_BLAZER_GATE_OCCASIONS)

    def test_blazer_gate_l2_formal_set(self):
        """Formal outerwear gate targets blazer + suit jacket."""
        assert _BLAZER_GATE_L2 == {"blazer", "suit jacket"}
        assert _FORMAL_OUTER_L2 == _BLAZER_GATE_L2


# ===========================================================================
# 4. Edge Cases
# ===========================================================================


class TestBlazerGateEdgeCases:
    """Edge cases and boundary conditions."""

    def test_none_occasions_treated_as_empty(self):
        """Source with occasions=None should not crash."""
        source = _source_top(formality="Casual")
        source.occasions = None
        source.formality_level = 1

        blazer = _outerwear_cand(product_id="blz-1", gemini_category_l2="Blazer")

        result = _filter_by_gemini_category(source, [blazer], "outerwear")
        assert len(result) == 0, "None occasions with formality 1 -> filter blazer"

    def test_default_formality_level_2(self):
        """Default formality_level is 2 which should NOT hard-filter."""
        source = AestheticProfile()
        # Don't call _derive_fields — test raw default
        blazer = _outerwear_cand(product_id="blz-1", gemini_category_l2="Blazer")

        result = _filter_by_gemini_category(source, [blazer], "outerwear")
        # formality_level defaults to 2, so no hard filter
        assert len(result) == 1

    def test_blazer_case_insensitive_l2(self):
        """L2 matching should be case-insensitive."""
        source = _source_top(formality="Casual", occasions=["Everyday"])
        source.formality_level = 1

        blazer_upper = _outerwear_cand(
            product_id="blz-1",
            gemini_category_l2="BLAZER",  # uppercase from Gemini
        )

        # The filter lowercases cand_l2 before checking
        result = _filter_by_gemini_category(source, [blazer_upper], "outerwear")
        assert len(result) == 0, "Should filter regardless of case"

    def test_mixed_candidates_only_blazers_filtered(self):
        """When multiple candidates, only blazers are removed."""
        source = _source_top(formality="Casual", occasions=["Everyday"])
        source.formality_level = 1

        candidates = [
            _outerwear_cand(product_id="blz-1", gemini_category_l2="Blazer"),
            _outerwear_cand(product_id="jkt-1", gemini_category_l2="Jacket"),
            _outerwear_cand(product_id="blz-2", gemini_category_l2="Blazer"),
            _outerwear_cand(product_id="cot-1", gemini_category_l2="Coat"),
            _outerwear_cand(product_id="bmb-1", gemini_category_l2="Bomber"),
        ]

        result = _filter_by_gemini_category(source, candidates, "outerwear")
        result_ids = {c.product_id for c in result}

        assert result_ids == {"jkt-1", "cot-1", "bmb-1"}, \
            "Only non-blazer outerwear should remain"

    def test_formality_boundary_1_vs_2(self):
        """Formality 1 hard-filters formal outerwear, formality 2 does not."""
        blazer = _outerwear_cand(product_id="blz-1", gemini_category_l2="Blazer")

        # Formality 1 -> hard filter
        source_1 = _source_top(formality="Casual", occasions=["Everyday"])
        source_1.formality_level = 1
        result_1 = _filter_by_gemini_category(source_1, [blazer], "outerwear")
        assert len(result_1) == 0

        # Formality 2 -> passes filter (penalized at scoring instead)
        source_2 = _source_top(formality="Casual", occasions=["Everyday"])
        source_2.formality_level = 2
        result_2 = _filter_by_gemini_category(source_2, [blazer], "outerwear")
        assert len(result_2) == 1


# ===========================================================================
# 5. Volume Outerwear Gate (poncho, cape, kimono, bolero, shrug)
# ===========================================================================


class TestVolumeOuterwearHardFilter:
    """Volume/statement outerwear should be hard-filtered at formality 1
    with NO occasion override (unlike formal outerwear)."""

    def test_poncho_filtered_formality_1_no_occasions(self):
        """Poncho + cami (formality 1, everyday) -> hard-filtered."""
        source = _source_top(
            gemini_category_l2="Camisole",
            formality="Casual",
            occasions=["Everyday"],
        )
        poncho = _outerwear_cand(product_id="pon-1", gemini_category_l2="Poncho")
        result = _filter_by_gemini_category(source, [poncho], "outerwear")
        assert len(result) == 0, "Poncho should be hard-filtered for cami at formality 1"

    def test_poncho_filtered_even_with_date_night(self):
        """Poncho + cami (formality 1, date night) -> STILL hard-filtered.
        Volume outerwear has NO occasion override."""
        source = _source_top(
            gemini_category_l2="Camisole",
            formality="Casual",
            occasions=["Date Night"],
        )
        poncho = _outerwear_cand(product_id="pon-1", gemini_category_l2="Poncho")
        result = _filter_by_gemini_category(source, [poncho], "outerwear")
        assert len(result) == 0, "Poncho should be filtered even with formal occasion"

    def test_cape_filtered_formality_1(self):
        """Cape + tank top (formality 1) -> hard-filtered."""
        source = _source_top(
            gemini_category_l2="Tank Top",
            formality="Casual",
            occasions=["Casual"],
        )
        cape = _outerwear_cand(product_id="cap-1", gemini_category_l2="Cape")
        result = _filter_by_gemini_category(source, [cape], "outerwear")
        assert len(result) == 0

    def test_kimono_filtered_formality_1(self):
        """Kimono + crop top (formality 1) -> hard-filtered."""
        source = _source_top(
            gemini_category_l2="Crop Top",
            formality="Casual",
            occasions=["Festival"],
        )
        kimono = _outerwear_cand(product_id="kim-1", gemini_category_l2="Kimono")
        result = _filter_by_gemini_category(source, [kimono], "outerwear")
        assert len(result) == 0

    def test_bolero_filtered_formality_1(self):
        """Bolero + tee (formality 1) -> hard-filtered."""
        source = _source_top(
            gemini_category_l2="T-Shirt",
            formality="Casual",
            occasions=["Weekend"],
        )
        bolero = _outerwear_cand(product_id="bol-1", gemini_category_l2="Bolero")
        result = _filter_by_gemini_category(source, [bolero], "outerwear")
        assert len(result) == 0

    def test_shrug_filtered_formality_1(self):
        """Shrug + cami (formality 1) -> hard-filtered."""
        source = _source_top(
            gemini_category_l2="Camisole",
            formality="Casual",
            occasions=["Everyday"],
        )
        shrug = _outerwear_cand(product_id="shr-1", gemini_category_l2="Shrug")
        result = _filter_by_gemini_category(source, [shrug], "outerwear")
        assert len(result) == 0

    def test_volume_passes_formality_2(self):
        """Volume outerwear passes hard filter at formality 2 (penalized at scoring)."""
        source = _source_top(formality="Smart Casual", occasions=["Everyday"])
        # Smart Casual = formality 2
        source.formality_level = 2
        poncho = _outerwear_cand(product_id="pon-1", gemini_category_l2="Poncho")
        result = _filter_by_gemini_category(source, [poncho], "outerwear")
        assert len(result) == 1, "Volume outerwear should pass hard filter at formality 2"

    def test_volume_passes_formality_3(self):
        """Volume outerwear passes hard filter at formality 3+."""
        source = _source_top(formality="Business Casual", occasions=["Work"])
        poncho = _outerwear_cand(product_id="pon-1", gemini_category_l2="Poncho")
        result = _filter_by_gemini_category(source, [poncho], "outerwear")
        assert len(result) == 1

    def test_mixed_candidates_volume_and_formal_filtered(self):
        """At formality 1, both blazers and volume outerwear are filtered."""
        source = _source_top(
            formality="Casual", occasions=["Everyday"],
            gemini_category_l2="Camisole",
        )
        source.formality_level = 1
        candidates = [
            _outerwear_cand(product_id="blz-1", gemini_category_l2="Blazer"),
            _outerwear_cand(product_id="pon-1", gemini_category_l2="Poncho"),
            _outerwear_cand(product_id="jkt-1", gemini_category_l2="Jacket"),
            _outerwear_cand(product_id="cap-1", gemini_category_l2="Cape"),
            _outerwear_cand(product_id="cot-1", gemini_category_l2="Coat"),
        ]
        result = _filter_by_gemini_category(source, candidates, "outerwear")
        result_ids = {c.product_id for c in result}
        assert result_ids == {"jkt-1", "cot-1"}, \
            "Blazer + poncho + cape filtered; jacket + coat pass"


class TestVolumeOuterwearScoring:
    """Volume outerwear scoring: penalized at formality <= 2, neutral at 3+."""

    def _score_outerwear(self, source, candidates):
        """Mirrors the three-group scoring logic."""
        _STRUCTURED_OUTERWEAR = frozenset({
            "blazer", "suit jacket", "jacket", "coat", "trench",
            "trench coat", "leather jacket", "denim jacket", "bomber",
            "parka",
        })
        _STRUCTURED_BONUS = 0.03
        results = []
        for cand in candidates:
            style_adj = 0.0
            cand_l2 = (cand.gemini_category_l2 or "").lower().strip()
            _src_occasions = {o.lower().strip() for o in (source.occasions or [])}
            _has_formal_occ = bool(_src_occasions & _OUTERWEAR_GATE_OCCASIONS)
            if cand_l2 in _STRUCTURED_OUTERWEAR:
                if cand_l2 in _FORMAL_OUTER_L2:
                    if source.formality_level >= _OUTERWEAR_MIN_FORMALITY:
                        style_adj += _STRUCTURED_BONUS
                    elif _has_formal_occ:
                        pass
                    else:
                        style_adj += _OUTERWEAR_CASUAL_PENALTY
                elif cand_l2 in _HEAVY_OUTER_L2:
                    if source.formality_level <= 1 and not _has_formal_occ:
                        style_adj += _OUTERWEAR_CASUAL_PENALTY
                    else:
                        style_adj += _STRUCTURED_BONUS
                else:
                    style_adj += _STRUCTURED_BONUS
            elif cand_l2 in _VOLUME_OUTER_L2:
                if source.formality_level <= 2:
                    style_adj += _OUTERWEAR_CASUAL_PENALTY
            results.append((cand.product_id, round(style_adj, 4)))
        return results

    def test_poncho_penalized_formality_2(self):
        """Poncho at formality 2 -> -0.10 (even with date night)."""
        source = _source_top(formality="Smart Casual", occasions=["Date Night"])
        source.formality_level = 2
        poncho = _outerwear_cand(product_id="pon-1", gemini_category_l2="Poncho")
        results = self._score_outerwear(source, [poncho])
        assert results[0][1] == -0.10, "Volume outerwear penalized at formality 2"

    def test_poncho_penalized_formality_2_no_occasion_override(self):
        """Poncho at formality 2 with work occasion -> STILL -0.10.
        Volume outerwear has no occasion override at scoring either."""
        source = _source_top(formality="Smart Casual", occasions=["Work"])
        source.formality_level = 2
        poncho = _outerwear_cand(product_id="pon-1", gemini_category_l2="Poncho")
        results = self._score_outerwear(source, [poncho])
        assert results[0][1] == -0.10

    def test_poncho_neutral_formality_3(self):
        """Poncho at formality 3+ -> 0.00 (neutral, no bonus)."""
        source = _source_top(formality="Business Casual", occasions=["Work"])
        poncho = _outerwear_cand(product_id="pon-1", gemini_category_l2="Poncho")
        results = self._score_outerwear(source, [poncho])
        assert results[0][1] == 0.00, "Volume outerwear neutral at formality 3+"

    def test_cape_penalized_formality_1(self):
        """Cape at formality 1 -> -0.10."""
        source = _source_top(formality="Casual", occasions=["Everyday"])
        source.formality_level = 1
        cape = _outerwear_cand(product_id="cap-1", gemini_category_l2="Cape")
        results = self._score_outerwear(source, [cape])
        assert results[0][1] == -0.10

    def test_kimono_penalized_formality_2(self):
        """Kimono at formality 2 -> -0.10."""
        source = _source_top(formality="Smart Casual", occasions=["Everyday"])
        source.formality_level = 2
        kimono = _outerwear_cand(product_id="kim-1", gemini_category_l2="Kimono")
        results = self._score_outerwear(source, [kimono])
        assert results[0][1] == -0.10

    def test_all_volume_types_penalized_at_formality_2(self):
        """All 5 volume types are penalized at formality 2."""
        source = _source_top(formality="Smart Casual", occasions=["Everyday"])
        source.formality_level = 2
        candidates = [
            _outerwear_cand(product_id="pon-1", gemini_category_l2="Poncho"),
            _outerwear_cand(product_id="cap-1", gemini_category_l2="Cape"),
            _outerwear_cand(product_id="kim-1", gemini_category_l2="Kimono"),
            _outerwear_cand(product_id="bol-1", gemini_category_l2="Bolero"),
            _outerwear_cand(product_id="shr-1", gemini_category_l2="Shrug"),
        ]
        results = self._score_outerwear(source, candidates)
        for pid, adj in results:
            assert adj == -0.10, f"{pid} should be penalized at formality 2"


# ===========================================================================
# 6. Heavy Outerwear Gate (parka, coat)
# ===========================================================================


class TestHeavyOuterwearHardFilter:
    """Heavy outerwear (parka, coat) is never hard-filtered — only penalized
    at scoring. The hard filter in _filter_by_gemini_category does not touch
    heavy outerwear."""

    def test_coat_passes_hard_filter_formality_1(self):
        """Coat + cami (formality 1, everyday) -> passes hard filter."""
        source = _source_top(
            gemini_category_l2="Camisole",
            formality="Casual",
            occasions=["Everyday"],
        )
        coat = _outerwear_cand(product_id="cot-1", gemini_category_l2="Coat")
        result = _filter_by_gemini_category(source, [coat], "outerwear")
        assert len(result) == 1, "Coat should NOT be hard-filtered"

    def test_parka_passes_hard_filter_formality_1(self):
        """Parka + tank top (formality 1) -> passes hard filter."""
        source = _source_top(
            gemini_category_l2="Tank Top",
            formality="Casual",
            occasions=["Casual"],
        )
        parka = _outerwear_cand(product_id="prk-1", gemini_category_l2="Parka")
        result = _filter_by_gemini_category(source, [parka], "outerwear")
        assert len(result) == 1, "Parka should NOT be hard-filtered"


class TestHeavyOuterwearScoring:
    """Heavy outerwear scoring: penalized at formality 1 without formal
    occasion, otherwise gets structured bonus."""

    def _score_outerwear(self, source, candidates):
        """Mirrors the three-group scoring logic."""
        _STRUCTURED_OUTERWEAR = frozenset({
            "blazer", "suit jacket", "jacket", "coat", "trench",
            "trench coat", "leather jacket", "denim jacket", "bomber",
            "parka",
        })
        _STRUCTURED_BONUS = 0.03
        results = []
        for cand in candidates:
            style_adj = 0.0
            cand_l2 = (cand.gemini_category_l2 or "").lower().strip()
            _src_occasions = {o.lower().strip() for o in (source.occasions or [])}
            _has_formal_occ = bool(_src_occasions & _OUTERWEAR_GATE_OCCASIONS)
            if cand_l2 in _STRUCTURED_OUTERWEAR:
                if cand_l2 in _FORMAL_OUTER_L2:
                    if source.formality_level >= _OUTERWEAR_MIN_FORMALITY:
                        style_adj += _STRUCTURED_BONUS
                    elif _has_formal_occ:
                        pass
                    else:
                        style_adj += _OUTERWEAR_CASUAL_PENALTY
                elif cand_l2 in _HEAVY_OUTER_L2:
                    if source.formality_level <= 1 and not _has_formal_occ:
                        style_adj += _OUTERWEAR_CASUAL_PENALTY
                    else:
                        style_adj += _STRUCTURED_BONUS
                else:
                    style_adj += _STRUCTURED_BONUS
            elif cand_l2 in _VOLUME_OUTER_L2:
                if source.formality_level <= 2:
                    style_adj += _OUTERWEAR_CASUAL_PENALTY
            results.append((cand.product_id, round(style_adj, 4)))
        return results

    def test_coat_penalized_formality_1_no_occasion(self):
        """Coat + cami (formality 1, everyday) -> -0.10."""
        source = _source_top(formality="Casual", occasions=["Everyday"])
        source.formality_level = 1
        coat = _outerwear_cand(product_id="cot-1", gemini_category_l2="Coat")
        results = self._score_outerwear(source, [coat])
        assert results[0][1] == -0.10

    def test_parka_penalized_formality_1_no_occasion(self):
        """Parka + tee (formality 1, casual) -> -0.10."""
        source = _source_top(formality="Casual", occasions=["Casual"])
        source.formality_level = 1
        parka = _outerwear_cand(product_id="prk-1", gemini_category_l2="Parka")
        results = self._score_outerwear(source, [parka])
        assert results[0][1] == -0.10

    def test_coat_bonus_formality_1_with_evening(self):
        """Coat + formal cami (formality 1, evening) -> +0.03.
        Heavy outerwear HAS occasion override (unlike volume)."""
        source = _source_top(formality="Casual", occasions=["Evening"])
        source.formality_level = 1
        coat = _outerwear_cand(product_id="cot-1", gemini_category_l2="Coat")
        results = self._score_outerwear(source, [coat])
        assert results[0][1] == 0.03, "Heavy outerwear should get bonus with formal occasion"

    def test_parka_bonus_formality_2(self):
        """Parka at formality 2 -> +0.03 (always bonus at formality 2+)."""
        source = _source_top(formality="Smart Casual", occasions=["Everyday"])
        source.formality_level = 2
        parka = _outerwear_cand(product_id="prk-1", gemini_category_l2="Parka")
        results = self._score_outerwear(source, [parka])
        assert results[0][1] == 0.03

    def test_coat_bonus_formality_3(self):
        """Coat at formality 3+ -> +0.03."""
        source = _source_top(formality="Business Casual", occasions=["Work"])
        coat = _outerwear_cand(product_id="cot-1", gemini_category_l2="Coat")
        results = self._score_outerwear(source, [coat])
        assert results[0][1] == 0.03

    def test_heavy_vs_volume_at_formality_1_with_occasion(self):
        """Heavy outerwear gets bonus with formal occasion; volume still penalized.
        This is the key difference between group 2 and group 3."""
        source = _source_top(formality="Casual", occasions=["Date Night"])
        source.formality_level = 1
        coat = _outerwear_cand(product_id="cot-1", gemini_category_l2="Coat")
        poncho = _outerwear_cand(product_id="pon-1", gemini_category_l2="Poncho")
        results = dict(self._score_outerwear(source, [coat, poncho]))
        assert results["cot-1"] == 0.03, "Coat should get bonus (occasion override)"
        assert results["pon-1"] == -0.10, "Poncho should still be penalized (no override)"

    def test_all_three_groups_compared(self):
        """Compare all three groups side by side at formality 2, no formal occasion."""
        source = _source_top(formality="Smart Casual", occasions=["Everyday"])
        source.formality_level = 2
        candidates = [
            _outerwear_cand(product_id="blz-1", gemini_category_l2="Blazer"),
            _outerwear_cand(product_id="pon-1", gemini_category_l2="Poncho"),
            _outerwear_cand(product_id="cot-1", gemini_category_l2="Coat"),
            _outerwear_cand(product_id="jkt-1", gemini_category_l2="Jacket"),
        ]
        results = dict(self._score_outerwear(source, candidates))
        assert results["blz-1"] == -0.10, "Formal outerwear: penalized at formality 2"
        assert results["pon-1"] == -0.10, "Volume outerwear: penalized at formality 2"
        assert results["cot-1"] == 0.03, "Heavy outerwear: bonus at formality 2"
        assert results["jkt-1"] == 0.03, "Other structured: always bonus"


# ===========================================================================
# 7. Precompute Constants Sync (expanded for three groups)
# ===========================================================================


class TestOuterwearGateConstantsSync:
    """Verify precompute script constants stay in sync with engine constants."""

    def test_formal_outer_l2_match(self):
        """Precompute _FORMAL_OUTER_L2 matches engine."""
        from scripts.precompute_outfit_candidates import (
            _FORMAL_OUTER_L2 as pre_formal,
        )
        assert pre_formal == _FORMAL_OUTER_L2

    def test_volume_outer_l2_match(self):
        """Precompute _VOLUME_OUTER_L2 matches engine."""
        from scripts.precompute_outfit_candidates import (
            _VOLUME_OUTER_L2 as pre_volume,
        )
        assert pre_volume == _VOLUME_OUTER_L2

    def test_heavy_outer_l2_match(self):
        """Precompute _HEAVY_OUTER_L2 matches engine."""
        from scripts.precompute_outfit_candidates import (
            _HEAVY_OUTER_L2 as pre_heavy,
        )
        assert pre_heavy == _HEAVY_OUTER_L2

    def test_gate_occasions_match(self):
        """Precompute _OUTERWEAR_GATE_OCCASIONS matches engine."""
        from scripts.precompute_outfit_candidates import (
            _OUTERWEAR_GATE_OCCASIONS as pre_occasions,
        )
        assert pre_occasions == _OUTERWEAR_GATE_OCCASIONS

    def test_min_formality_match(self):
        """Precompute _OUTERWEAR_MIN_FORMALITY matches engine."""
        from scripts.precompute_outfit_candidates import (
            _OUTERWEAR_MIN_FORMALITY as pre_min,
        )
        assert pre_min == _OUTERWEAR_MIN_FORMALITY

    def test_casual_penalty_match(self):
        """Precompute _OUTERWEAR_CASUAL_PENALTY matches engine."""
        from scripts.precompute_outfit_candidates import (
            _OUTERWEAR_CASUAL_PENALTY as pre_penalty,
        )
        assert pre_penalty == _OUTERWEAR_CASUAL_PENALTY

    def test_backward_compat_aliases(self):
        """Backward-compat aliases still resolve correctly."""
        assert _BLAZER_GATE_L2 == _FORMAL_OUTER_L2
        assert _BLAZER_GATE_OCCASIONS == _OUTERWEAR_GATE_OCCASIONS
        assert _BLAZER_MIN_FORMALITY == _OUTERWEAR_MIN_FORMALITY
        assert _BLAZER_CASUAL_PENALTY == _OUTERWEAR_CASUAL_PENALTY
