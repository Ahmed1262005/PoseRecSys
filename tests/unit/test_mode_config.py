"""
Unit tests for the mode-based configuration system.

Tests expand_modes(), mode implications, conflict detection,
RRF weight lookup, and the mode menu text generation.
"""

import logging
import pytest

from search.mode_config import (
    MODE_CONFIGS,
    MODE_CONFLICTS,
    MODE_IMPLIES,
    VALID_MODES,
    ModeConfig,
    expand_modes,
    get_mode_menu_text,
    get_rrf_weights,
)


# ---------------------------------------------------------------------------
# TestModeExpansion — core expand_modes() tests
# ---------------------------------------------------------------------------


class TestModeExpansion:
    """Tests for the expand_modes() function."""

    def test_empty_modes_returns_empty(self):
        """Empty list should return empty dicts."""
        filters, exclusions, expanded, name_excl = expand_modes([])
        assert filters == {}
        assert exclusions == {}
        assert expanded == {}
        assert name_excl == []

    def test_single_coverage_mode(self):
        """A single coverage mode should produce the correct exclusions."""
        filters, exclusions, expanded, _ = expand_modes(["cover_arms"])

        # cover_arms has no positive filters
        assert filters == {}

        # Should exclude revealing sleeves, necklines, and sheer materials
        assert "sleeve_type" in exclusions
        assert "Sleeveless" in exclusions["sleeve_type"]
        assert "Short" in exclusions["sleeve_type"]
        assert "Cap" in exclusions["sleeve_type"]
        assert "Spaghetti Strap" in exclusions["sleeve_type"]

        assert "neckline" in exclusions
        assert "Off-Shoulder" in exclusions["neckline"]
        assert "Strapless" in exclusions["neckline"]

        assert "materials" in exclusions
        assert "Mesh" in exclusions["materials"]
        assert "Lace" in exclusions["materials"]

    def test_cover_chest_exclusions(self):
        """cover_chest should exclude revealing necklines."""
        _, exclusions, _, _ = expand_modes(["cover_chest"])
        assert "neckline" in exclusions
        for neckline in ["V-Neck", "Deep V-Neck", "Sweetheart", "Halter",
                         "Off-Shoulder", "Strapless", "Plunging"]:
            assert neckline in exclusions["neckline"], f"Missing {neckline}"

    def test_cover_legs_exclusions(self):
        """cover_legs should exclude short lengths."""
        _, exclusions, _, _ = expand_modes(["cover_legs"])
        assert "length" in exclusions
        assert "Mini" in exclusions["length"]
        assert "Cropped" in exclusions["length"]

    def test_opaque_exclusions(self):
        """opaque should exclude sheer/transparent materials."""
        _, exclusions, _, _ = expand_modes(["opaque"])
        assert "materials" in exclusions
        for mat in ["Mesh", "Lace", "Chiffon", "Sheer"]:
            assert mat in exclusions["materials"]

    def test_occasion_mode_produces_filters(self):
        """Occasion modes should produce positive filters."""
        filters, exclusions, _, _ = expand_modes(["work"])
        assert "occasions" in filters
        assert "Office" in filters["occasions"]
        assert "formality" in filters
        assert "Business Casual" in filters["formality"]
        # No exclusions for a pure occasion mode
        assert exclusions == {}

    def test_aesthetic_mode_produces_style_tags(self):
        """Aesthetic modes should produce style_tags filters."""
        filters, _, _, _ = expand_modes(["quiet_luxury"])
        assert "style_tags" in filters
        assert "Classic" in filters["style_tags"]
        assert "Minimalist" in filters["style_tags"]
        assert "formality" in filters
        assert "Smart Casual" in filters["formality"]

    def test_multiple_modes_union(self):
        """Multiple modes should union their filters without duplicates."""
        filters, exclusions, _, _ = expand_modes(["cover_arms", "cover_chest"])

        # Both produce neckline exclusions — should be merged
        assert "neckline" in exclusions
        necklines = exclusions["neckline"]
        # From cover_arms
        assert "Off-Shoulder" in necklines
        assert "Strapless" in necklines
        # From cover_chest
        assert "V-Neck" in necklines
        assert "Sweetheart" in necklines
        # No duplicates
        assert len(necklines) == len(set(necklines))

    def test_weather_mode(self):
        """Weather modes should set season filters."""
        filters, _, _, _ = expand_modes(["hot_weather"])
        assert "seasons" in filters
        assert "Summer" in filters["seasons"]

    def test_formality_mode(self):
        """Formality modes should set formality filters."""
        filters, _, _, _ = expand_modes(["smart_casual"])
        assert "formality" in filters
        assert "Smart Casual" in filters["formality"]

    def test_unknown_mode_ignored(self, caplog):
        """Unknown mode names should be ignored with a warning."""
        with caplog.at_level(logging.WARNING):
            filters, exclusions, _, _ = expand_modes(["nonexistent_mode"])

        assert filters == {}
        assert exclusions == {}
        assert "Unknown mode 'nonexistent_mode'" in caplog.text

    def test_unknown_mode_mixed_with_valid(self, caplog):
        """Unknown modes ignored while valid modes still work."""
        with caplog.at_level(logging.WARNING):
            filters, exclusions, _, _ = expand_modes(["nonexistent", "cover_arms"])

        # cover_arms should still work
        assert "sleeve_type" in exclusions
        assert "Unknown mode 'nonexistent'" in caplog.text

    def test_duplicate_modes_handled(self):
        """Duplicate mode names should not cause duplicate values."""
        filters, exclusions, _, _ = expand_modes(["cover_arms", "cover_arms"])
        if "sleeve_type" in exclusions:
            assert len(exclusions["sleeve_type"]) == len(set(exclusions["sleeve_type"]))

    def test_expanded_filters_mirrors_mode_filters(self):
        """expanded_filters should contain the same values as mode_filters."""
        filters, _, expanded, _ = expand_modes(["quiet_luxury", "date_night"])
        assert filters == expanded
        # But they should be separate objects (copies)
        assert filters is not expanded

    def test_occasion_plus_coverage_combined(self):
        """Combining occasion + coverage modes should produce both filters and exclusions."""
        filters, exclusions, _, _ = expand_modes(["wedding_guest", "cover_chest"])

        # From wedding_guest
        assert "occasions" in filters
        assert "Wedding Guest" in filters["occasions"]
        assert "formality" in filters

        # From cover_chest
        assert "neckline" in exclusions
        assert "V-Neck" in exclusions["neckline"]


# ---------------------------------------------------------------------------
# TestModeImplications — implication chain tests
# ---------------------------------------------------------------------------


class TestModeImplications:
    """Tests for mode implication resolution."""

    def test_modest_implies_all_coverage_and_opaque(self):
        """modest should expand to all coverage modes + opaque."""
        filters, exclusions, _, name_excl = expand_modes(["modest"])

        # Should have exclusions from all coverage modes
        assert "sleeve_type" in exclusions  # from cover_arms
        assert "neckline" in exclusions     # from cover_arms + cover_chest + cover_straps
        assert "style_tags" in exclusions   # from cover_back + cover_straps
        assert "length" in exclusions       # from cover_legs + cover_stomach
        assert "fit_type" in exclusions     # from cover_stomach
        assert "silhouette" in exclusions   # from cover_stomach
        assert "materials" in exclusions    # from opaque + cover_arms

        # Specific values
        assert "Sleeveless" in exclusions["sleeve_type"]
        assert "V-Neck" in exclusions["neckline"]
        assert "Backless" in exclusions["style_tags"]
        assert "Mini" in exclusions["length"]
        assert "Bodycon" in exclusions["silhouette"]
        assert "Sheer" in exclusions["materials"]

    def test_funeral_implies_modest(self):
        """funeral -> modest -> all coverage + opaque (transitive)."""
        f_funeral, e_funeral, _, _ = expand_modes(["funeral"])
        f_modest, e_modest, _, _ = expand_modes(["modest"])

        # funeral exclusions should be a superset of modest exclusions
        for key in e_modest:
            assert key in e_funeral, f"Missing exclusion key: {key}"
            for val in e_modest[key]:
                assert val in e_funeral[key], f"Missing {key}={val}"

        # funeral also adds formality filter
        assert "formality" in f_funeral
        assert "Formal" in f_funeral["formality"]

    def test_religious_event_implies_modest(self):
        """religious_event -> modest -> all coverage + opaque."""
        f_rel, e_rel, _, _ = expand_modes(["religious_event"])
        _, e_modest, _, _ = expand_modes(["modest"])

        # Should have all modest exclusions
        for key in e_modest:
            assert key in e_rel, f"Missing exclusion key: {key}"
            for val in e_modest[key]:
                assert val in e_rel[key], f"Missing {key}={val}"

        # Plus formality filter
        assert "formality" in f_rel
        assert "Formal" in f_rel["formality"]

    def test_no_circular_implications(self):
        """Ensure implications don't create infinite loops."""
        # Every mode in MODE_IMPLIES values should either not be in
        # MODE_IMPLIES or not create a cycle back to the key
        visited: set = set()

        def check_no_cycle(mode: str, path: list):
            if mode in path:
                pytest.fail(f"Circular implication detected: {' -> '.join(path + [mode])}")
            path.append(mode)
            for implied in MODE_IMPLIES.get(mode, []):
                if implied not in visited:
                    check_no_cycle(implied, path.copy())
            visited.add(mode)

        for mode in MODE_IMPLIES:
            check_no_cycle(mode, [])

    def test_implication_targets_are_valid_modes(self):
        """All modes referenced in MODE_IMPLIES should be valid modes."""
        for source, targets in MODE_IMPLIES.items():
            assert source in VALID_MODES, f"Source mode '{source}' not in VALID_MODES"
            for target in targets:
                assert target in VALID_MODES, f"Implied mode '{target}' not in VALID_MODES"


# ---------------------------------------------------------------------------
# TestModeConflicts — conflict detection tests
# ---------------------------------------------------------------------------


class TestModeConflicts:
    """Tests for mode conflict detection."""

    def test_relaxed_fit_not_oversized_conflict(self, caplog):
        """relaxed_fit + not_oversized should log a warning."""
        with caplog.at_level(logging.WARNING):
            expand_modes(["relaxed_fit", "not_oversized"])
        assert "Conflicting modes" in caplog.text

    def test_non_conflicting_modes_no_warning(self, caplog):
        """Non-conflicting modes should not produce a conflict warning."""
        with caplog.at_level(logging.WARNING):
            expand_modes(["cover_arms", "work"])
        assert "Conflicting modes" not in caplog.text

    def test_conflict_does_not_crash(self):
        """Conflicting modes should still produce output (both applied)."""
        filters, exclusions, _, _ = expand_modes(["relaxed_fit", "not_oversized"])
        # Both exclusions should be present
        assert "fit_type" in exclusions
        assert "Fitted" in exclusions["fit_type"]    # from relaxed_fit
        assert "Oversized" in exclusions["fit_type"]  # from not_oversized

    def test_conflict_pairs_reference_valid_modes(self):
        """All modes in conflict pairs should be valid modes."""
        for pair in MODE_CONFLICTS:
            for mode in pair:
                assert mode in VALID_MODES, f"Conflict mode '{mode}' not in VALID_MODES"


# ---------------------------------------------------------------------------
# TestRRFWeights — weight lookup tests
# ---------------------------------------------------------------------------


class TestRRFWeights:
    """Tests for get_rrf_weights()."""

    def test_exact_weights(self):
        """Exact intent should heavily favor Algolia."""
        algolia_w, semantic_w = get_rrf_weights("exact")
        assert algolia_w == pytest.approx(0.85)
        assert semantic_w == pytest.approx(0.15)
        assert algolia_w + semantic_w == pytest.approx(1.0)

    def test_specific_weights(self):
        """Specific intent should balance Algolia and semantic."""
        algolia_w, semantic_w = get_rrf_weights("specific")
        assert algolia_w == pytest.approx(0.60)
        assert semantic_w == pytest.approx(0.40)

    def test_vague_weights(self):
        """Vague intent should favor semantic search."""
        algolia_w, semantic_w = get_rrf_weights("vague")
        assert algolia_w == pytest.approx(0.35)
        assert semantic_w == pytest.approx(0.65)

    def test_unknown_intent_defaults_to_specific(self):
        """Unknown intent should default to specific weights."""
        algolia_w, semantic_w = get_rrf_weights("unknown_intent")
        assert algolia_w == pytest.approx(0.60)
        assert semantic_w == pytest.approx(0.40)

    def test_weights_sum_to_one(self):
        """All weight pairs should sum to 1.0."""
        for intent in ["exact", "specific", "vague"]:
            a, s = get_rrf_weights(intent)
            assert a + s == pytest.approx(1.0), f"{intent}: {a} + {s} != 1.0"


# ---------------------------------------------------------------------------
# TestModeMenuText — prompt generation tests
# ---------------------------------------------------------------------------


class TestModeMenuText:
    """Tests for get_mode_menu_text()."""

    def test_menu_contains_all_categories(self):
        """Menu text should contain all mode category headers."""
        text = get_mode_menu_text()
        assert "COVERAGE" in text
        assert "FIT" in text
        assert "OCCASION" in text
        assert "FORMALITY" in text
        assert "AESTHETIC" in text
        assert "WEATHER" in text

    def test_menu_contains_key_modes(self):
        """Menu text should reference important modes."""
        text = get_mode_menu_text()
        for mode in ["cover_arms", "modest", "work", "date_night",
                      "quiet_luxury", "relaxed_fit", "hot_weather"]:
            assert mode in text, f"Mode '{mode}' missing from menu text"


# ---------------------------------------------------------------------------
# TestModeConfigIntegrity — structural validation
# ---------------------------------------------------------------------------


class TestModeConfigIntegrity:
    """Structural integrity tests for the mode configuration."""

    def test_all_modes_have_config(self):
        """Every valid mode should have a ModeConfig entry."""
        for mode in VALID_MODES:
            assert mode in MODE_CONFIGS, f"Mode '{mode}' missing from MODE_CONFIGS"

    def test_config_values_are_lists_of_strings(self):
        """All filter/exclusion values should be lists of strings."""
        for name, config in MODE_CONFIGS.items():
            for key, values in config.filters.items():
                assert isinstance(values, list), f"{name}.filters[{key}] not a list"
                for v in values:
                    assert isinstance(v, str), f"{name}.filters[{key}] contains non-string: {v}"
            for key, values in config.exclusions.items():
                assert isinstance(values, list), f"{name}.exclusions[{key}] not a list"
                for v in values:
                    assert isinstance(v, str), f"{name}.exclusions[{key}] contains non-string: {v}"

    def test_cover_back_has_name_exclusions(self):
        """cover_back should include name exclusions for backless/open-back."""
        _, _, _, name_excl = expand_modes(["cover_back"])
        assert "backless" in name_excl
        assert "open back" in name_excl
        assert "open-back" in name_excl

    def test_cover_straps_has_name_exclusions(self):
        """cover_straps should include name exclusions for backless/open-back."""
        _, _, _, name_excl = expand_modes(["cover_straps"])
        assert "backless" in name_excl

    def test_modest_inherits_name_exclusions(self):
        """modest implies cover_back + cover_straps, so gets their name_exclusions."""
        _, _, _, name_excl = expand_modes(["modest"])
        assert "backless" in name_excl
        assert "open back" in name_excl

    def test_modes_without_name_exclusions(self):
        """Most modes have no name_exclusions."""
        _, _, _, name_excl = expand_modes(["work"])
        assert name_excl == []

    def test_mode_count(self):
        """Should have approximately 40 modes."""
        assert len(MODE_CONFIGS) >= 38, f"Only {len(MODE_CONFIGS)} modes, expected ~40+"
        assert len(MODE_CONFIGS) <= 50, f"{len(MODE_CONFIGS)} modes — unexpectedly many"
