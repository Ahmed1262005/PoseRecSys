"""
Unit tests for precomputed outfit candidate pool integration (v3.4).

Tests _fetch_precomputed_pool(), the precomputed-first path in
_score_category(), fallback to live retrieval, settings toggle,
and retrieval_strategies tracking in scoring_info.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from types import SimpleNamespace

from services.outfit_engine import (
    AestheticProfile,
    OutfitEngine,
    _derive_fields,
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


def _mock_rpc_row(candidate_id: str, similarity: float = 0.85, **kwargs):
    """Build a row matching the get_outfit_candidates RPC response shape."""
    defaults = {
        "candidate_id": candidate_id,
        "cosine_similarity": similarity,
        "rank": 1,
        "name": f"Product {candidate_id[:8]}",
        "brand": "TestBrand",
        "category": "Bottoms",
        "broad_category": "bottoms",
        "price": 55.0,
        "primary_image_url": f"https://img.test/{candidate_id}.jpg",
        "gallery_images": None,
        "colors": ["blue"],
        "materials": ["denim"],
        "gemini_category_l1": "Bottoms",
        "gemini_category_l2": "Straight Leg Jeans",
        "gemini_occasions": ["Everyday"],
        "gemini_style_tags": ["Classic"],
        "gemini_pattern": "Solid",
        "gemini_formality": "Smart Casual",
        "gemini_fit_type": "Regular",
        "gemini_color_family": "Blues",
        "gemini_primary_color": "Blue",
        "gemini_secondary_colors": None,
        "gemini_seasons": ["Spring", "Fall"],
        "gemini_silhouette": "Straight",
        "gemini_construction": None,
        "gemini_apparent_fabric": "Denim",
        "gemini_texture": "Textured",
        "gemini_coverage_level": "Full",
        "gemini_sheen": None,
        "gemini_rise": "Mid",
        "gemini_leg_shape": "Straight",
        "gemini_stretch": None,
        "gemini_styling_metadata": None,
        "gemini_styling_role": None,
        "gemini_appearance_top_tags": None,
        "gemini_vibe_tags": None,
        "gemini_extractor_version": "v1.0.0.2",
    }
    defaults.update(kwargs)
    return defaults


def _make_engine():
    """Create an OutfitEngine with a mocked Supabase client."""
    mock_supabase = MagicMock()
    engine = OutfitEngine(mock_supabase)
    return engine, mock_supabase


# ---------------------------------------------------------------------------
# _fetch_precomputed_pool
# ---------------------------------------------------------------------------

class TestFetchPrecomputedPool:
    """Tests for _fetch_precomputed_pool() method."""

    def test_returns_profiles_on_success(self):
        """When RPC returns rows, should return a list of AestheticProfiles."""
        engine, mock_sb = _make_engine()

        rows = [
            _mock_rpc_row("cand-001", similarity=0.90, rank=1),
            _mock_rpc_row("cand-002", similarity=0.85, rank=2),
            _mock_rpc_row("cand-003", similarity=0.80, rank=3),
        ]
        mock_sb.rpc.return_value.execute.return_value = SimpleNamespace(data=rows)

        result = engine._fetch_precomputed_pool("src-001", "bottoms", limit=60)

        assert result is not None
        assert len(result) == 3
        assert all(isinstance(p, AestheticProfile) for p in result)
        # Check product_ids are set from candidate_id
        pids = {p.product_id for p in result}
        assert pids == {"cand-001", "cand-002", "cand-003"}

    def test_similarity_mapped_correctly(self):
        """cosine_similarity from RPC should be mapped to profile.similarity."""
        engine, mock_sb = _make_engine()

        rows = [_mock_rpc_row("cand-001", similarity=0.92, rank=1)]
        mock_sb.rpc.return_value.execute.return_value = SimpleNamespace(data=rows)

        result = engine._fetch_precomputed_pool("src-001", "bottoms")

        assert result is not None
        assert abs(result[0].similarity - 0.92) < 0.001

    def test_gemini_attrs_parsed(self):
        """Gemini-prefixed columns should be parsed into AestheticProfile fields."""
        engine, mock_sb = _make_engine()

        rows = [_mock_rpc_row(
            "cand-001", rank=1,
            gemini_category_l1="Bottoms",
            gemini_category_l2="Wide Leg Pants",
            gemini_fit_type="Relaxed",
            gemini_formality="Casual",
            gemini_apparent_fabric="Linen",
        )]
        mock_sb.rpc.return_value.execute.return_value = SimpleNamespace(data=rows)

        result = engine._fetch_precomputed_pool("src-001", "bottoms")

        p = result[0]
        assert p.gemini_category_l1 == "Bottoms"
        assert p.gemini_category_l2 == "Wide Leg Pants"
        assert p.fit_type == "Relaxed"
        assert p.formality == "Casual"
        assert p.apparent_fabric == "Linen"

    def test_returns_none_on_empty_rows(self):
        """When RPC returns empty list, should return None (triggers fallback)."""
        engine, mock_sb = _make_engine()

        mock_sb.rpc.return_value.execute.return_value = SimpleNamespace(data=[])

        result = engine._fetch_precomputed_pool("src-001", "bottoms")

        assert result is None

    def test_returns_none_on_null_data(self):
        """When RPC returns null data, should return None."""
        engine, mock_sb = _make_engine()

        mock_sb.rpc.return_value.execute.return_value = SimpleNamespace(data=None)

        result = engine._fetch_precomputed_pool("src-001", "bottoms")

        assert result is None

    def test_returns_none_on_rpc_exception(self):
        """When RPC throws (table missing, timeout, etc.), should return None."""
        engine, mock_sb = _make_engine()

        mock_sb.rpc.side_effect = Exception("relation 'outfit_candidates' does not exist")

        result = engine._fetch_precomputed_pool("src-001", "bottoms")

        assert result is None

    def test_returns_none_on_execute_exception(self):
        """When execute() throws, should return None."""
        engine, mock_sb = _make_engine()

        mock_sb.rpc.return_value.execute.side_effect = Exception("statement timeout")

        result = engine._fetch_precomputed_pool("src-001", "bottoms")

        assert result is None

    def test_rpc_called_with_correct_params(self):
        """RPC should be called with correct parameter names and values."""
        engine, mock_sb = _make_engine()

        mock_sb.rpc.return_value.execute.return_value = SimpleNamespace(data=[])

        engine._fetch_precomputed_pool("abc-123", "outerwear", limit=30)

        mock_sb.rpc.assert_called_once_with("get_outfit_candidates", {
            "p_source_id": "abc-123",
            "p_target_category": "outerwear",
            "p_limit": 30,
        })

    def test_preserves_order_by_rank(self):
        """Profiles should preserve the rank ordering from the RPC."""
        engine, mock_sb = _make_engine()

        rows = [
            _mock_rpc_row("cand-a", similarity=0.95, rank=1),
            _mock_rpc_row("cand-b", similarity=0.90, rank=2),
            _mock_rpc_row("cand-c", similarity=0.85, rank=3),
        ]
        mock_sb.rpc.return_value.execute.return_value = SimpleNamespace(data=rows)

        result = engine._fetch_precomputed_pool("src-001", "tops")

        assert [p.product_id for p in result] == ["cand-a", "cand-b", "cand-c"]
        assert result[0].similarity > result[1].similarity > result[2].similarity


# ---------------------------------------------------------------------------
# Strategy P integration in _score_category
# ---------------------------------------------------------------------------

class TestPrecomputedPoolInScoreCategory:
    """Tests for precomputed pool integration in _score_category().

    These test the control flow: precomputed first, fallback to live,
    and the settings toggle. They mock both _fetch_precomputed_pool and
    the live retrieval paths to isolate the control flow logic.
    """

    def _make_candidates(self, n=5, target="bottoms"):
        """Create n candidate profiles for scoring."""
        candidates = []
        for i in range(n):
            p = _make_profile(
                product_id=f"cand-{i:03d}",
                broad_category=target,
                gemini_category_l1="Bottoms" if target == "bottoms" else "Outerwear",
                gemini_category_l2="Straight Leg Jeans" if target == "bottoms" else "Blazer",
                formality="Smart Casual",
                fit_type="Regular",
                occasions=["Everyday"],
                style_tags=["Classic"],
                color_family="Blues" if target == "bottoms" else "Blacks",
                apparent_fabric="Denim" if target == "bottoms" else "Wool",
                texture="Textured",
                pattern="Solid",
                silhouette="Straight" if target == "bottoms" else "Regular",
                length="Regular",
                coverage_level="Full",
                seasons=["Spring", "Fall"],
                price=50 + i * 5,
            )
            p.similarity = 0.90 - i * 0.02
            candidates.append(p)
        return candidates

    @patch("config.settings.get_settings")
    def test_precomputed_skips_live_retrieval(self, mock_settings):
        """When precomputed pool returns candidates, Strategy A/B should NOT run."""
        mock_settings.return_value = SimpleNamespace(
            use_precomputed_pools=True,
            use_styling_scorer=False,
            use_llm_judge=False,
        )
        engine, mock_sb = _make_engine()
        source = _source_top()
        candidates = self._make_candidates(10)

        with patch.object(engine, "_fetch_precomputed_pool", return_value=candidates) as mock_fetch, \
             patch.object(engine, "_encode_texts_batch") as mock_encode, \
             patch.object(engine, "_retrieve_candidates") as mock_retrieve:

            result = engine._score_category(
                source, "tops", "bottoms", "casual",
            )

            mock_fetch.assert_called_once_with(source.product_id, "bottoms", limit=60)
            mock_encode.assert_not_called()
            mock_retrieve.assert_not_called()
            assert len(result) > 0

    @patch("config.settings.get_settings")
    def test_fallback_when_precomputed_returns_none(self, mock_settings):
        """When precomputed returns None, should fall back to Strategy A/B."""
        mock_settings.return_value = SimpleNamespace(
            use_precomputed_pools=True,
            use_styling_scorer=False,
            use_llm_judge=False,
        )
        engine, mock_sb = _make_engine()
        source = _source_top()
        candidates = self._make_candidates(5)

        with patch.object(engine, "_fetch_precomputed_pool", return_value=None) as mock_fetch, \
             patch.object(engine, "_encode_texts_batch", side_effect=Exception("skip batch")), \
             patch.object(engine, "_retrieve_candidates", return_value=[]) as mock_pgvec, \
             patch.object(engine, "_retrieve_text_candidates", return_value=[]):

            result = engine._score_category(
                source, "tops", "bottoms", "casual",
            )

            mock_fetch.assert_called_once()
            # Live retrieval was attempted (even if it returned empty)
            mock_pgvec.assert_called_once()

    @patch("config.settings.get_settings")
    def test_toggle_off_skips_precomputed(self, mock_settings):
        """When use_precomputed_pools=False, should not call _fetch_precomputed_pool."""
        mock_settings.return_value = SimpleNamespace(
            use_precomputed_pools=False,
            use_styling_scorer=False,
            use_llm_judge=False,
        )
        engine, mock_sb = _make_engine()
        source = _source_top()

        with patch.object(engine, "_fetch_precomputed_pool") as mock_fetch, \
             patch.object(engine, "_encode_texts_batch", side_effect=Exception("skip batch")), \
             patch.object(engine, "_retrieve_candidates", return_value=[]), \
             patch.object(engine, "_retrieve_text_candidates", return_value=[]):

            engine._score_category(
                source, "tops", "bottoms", "casual",
            )

            mock_fetch.assert_not_called()

    @patch("config.settings.get_settings")
    def test_retrieval_strategy_tracked_precomputed(self, mock_settings):
        """_retrieval_strategies should record 'precomputed' when pool is used."""
        mock_settings.return_value = SimpleNamespace(
            use_precomputed_pools=True,
            use_styling_scorer=False,
            use_llm_judge=False,
        )
        engine, _ = _make_engine()
        source = _source_top()
        candidates = self._make_candidates(5)

        with patch.object(engine, "_fetch_precomputed_pool", return_value=candidates):
            engine._score_category(
                source, "tops", "bottoms", "casual",
            )

        assert engine._retrieval_strategies.get("bottoms") == "precomputed"

    @patch("config.settings.get_settings")
    def test_retrieval_strategy_tracked_fallback(self, mock_settings):
        """_retrieval_strategies should record 'multi_rpc' on fallback path."""
        mock_settings.return_value = SimpleNamespace(
            use_precomputed_pools=True,
            use_styling_scorer=False,
            use_llm_judge=False,
        )
        engine, _ = _make_engine()
        source = _source_top()

        # Precomputed returns None, batch fails, Strategy B returns some data
        mock_rows = [
            {
                "product_id": "cand-fallback",
                "name": "Fallback Jeans", "brand": "TestBrand",
                "category": "Bottoms", "broad_category": "bottoms",
                "price": 50, "primary_image_url": "https://img/x.jpg",
                "base_color": "blue", "similarity": 0.8,
                "category_l1": "Bottoms", "category_l2": "Jeans",
                "occasions": ["Everyday"], "style_tags": ["Classic"],
                "pattern": "Solid", "formality": "Smart Casual",
                "fit_type": "Regular", "color_family": "Blues",
                "primary_color": "Blue", "seasons": ["Spring"],
                "silhouette": "Straight", "apparent_fabric": "Denim",
                "texture": "Textured", "coverage_level": "Full",
            }
        ]

        with patch.object(engine, "_fetch_precomputed_pool", return_value=None), \
             patch.object(engine, "_encode_texts_batch", side_effect=Exception("skip")), \
             patch.object(engine, "_retrieve_candidates", return_value=mock_rows), \
             patch.object(engine, "_retrieve_text_candidates", return_value=[]):

            engine._score_category(
                source, "tops", "bottoms", "casual",
            )

        assert engine._retrieval_strategies.get("bottoms") == "multi_rpc"

    @patch("config.settings.get_settings")
    def test_precomputed_candidates_are_scored(self, mock_settings):
        """Precomputed candidates should go through the full TATTOO scoring pipeline."""
        mock_settings.return_value = SimpleNamespace(
            use_precomputed_pools=True,
            use_styling_scorer=False,
            use_llm_judge=False,
        )
        engine, _ = _make_engine()
        source = _source_top()
        candidates = self._make_candidates(3)

        with patch.object(engine, "_fetch_precomputed_pool", return_value=candidates):
            result = engine._score_category(
                source, "tops", "bottoms", "casual",
            )

        assert len(result) > 0
        for entry in result:
            assert "tattoo" in entry
            assert "compat" in entry
            assert "cosine" in entry
            assert "profile" in entry
            assert isinstance(entry["profile"], AestheticProfile)
            # TATTOO should be a weighted combination
            assert entry["tattoo"] > 0

    @patch("config.settings.get_settings")
    def test_precomputed_candidates_filtered_and_deduped(self, mock_settings):
        """Precomputed candidates should still pass through all filters."""
        mock_settings.return_value = SimpleNamespace(
            use_precomputed_pools=True,
            use_styling_scorer=False,
            use_llm_judge=False,
        )
        engine, _ = _make_engine()
        source = _source_top()

        # Create candidates with duplicates
        candidates = self._make_candidates(3)
        dupe = _make_profile(
            product_id="cand-000",  # same as first candidate
            broad_category="bottoms",
            gemini_category_l1="Bottoms",
            gemini_category_l2="Straight Leg Jeans",
            formality="Smart Casual",
            fit_type="Regular",
            pattern="Solid",
            silhouette="Straight",
            apparent_fabric="Denim",
            price=50,
        )
        dupe.similarity = 0.88
        candidates.append(dupe)

        with patch.object(engine, "_fetch_precomputed_pool", return_value=candidates):
            result = engine._score_category(
                source, "tops", "bottoms", "casual",
            )

        # Duplicate should be removed
        pids = [e["profile"].product_id for e in result]
        assert len(pids) == len(set(pids)), "Duplicates should be removed"


# ---------------------------------------------------------------------------
# build_outfit integration
# ---------------------------------------------------------------------------

class TestBuildOutfitPrecomputed:
    """Test that build_outfit correctly exposes retrieval_strategies in scoring_info."""

    @patch("config.settings.get_settings")
    def test_scoring_info_includes_retrieval_strategies(self, mock_settings):
        """scoring_info should contain retrieval_strategies dict."""
        mock_settings.return_value = SimpleNamespace(
            use_precomputed_pools=True,
            use_styling_scorer=False,
            use_llm_judge=False,
        )
        engine, _ = _make_engine()

        # Reset should clear strategies
        engine._retrieval_strategies = {"old": "data"}

        # Mock _fetch_product_with_attrs, _score_category, etc.
        source = _source_top()
        with patch.object(engine, "_fetch_product_with_attrs", return_value=source), \
             patch.object(engine, "_score_category", return_value=[]), \
             patch.object(engine, "_load_user_profile", return_value=None):

            result = engine.build_outfit("src-001")

        # The reset should have cleared old strategies
        assert "scoring_info" in result
        si = result["scoring_info"]
        assert "retrieval_strategies" in si
        # Old data should be gone (reset at start of build_outfit)
        assert si["retrieval_strategies"].get("old") is None

    def test_retrieval_strategies_reset_between_calls(self):
        """_retrieval_strategies should be cleared at the start of each build_outfit call."""
        engine, _ = _make_engine()

        # Simulate leftover state from previous call
        engine._retrieval_strategies = {"bottoms": "precomputed", "outerwear": "multi_rpc"}

        source = _source_top()
        with patch.object(engine, "_fetch_product_with_attrs", return_value=source), \
             patch.object(engine, "_score_category", return_value=[]), \
             patch.object(engine, "_load_user_profile", return_value=None), \
             patch("config.settings.get_settings") as mock_settings:
            mock_settings.return_value = SimpleNamespace(
                use_precomputed_pools=True,
                use_styling_scorer=False,
                use_llm_judge=False,
            )

            engine.build_outfit("src-001")

        # After build_outfit starts, old strategies should be cleared
        # (new ones would only be set by _score_category, which we mocked)
        assert engine._retrieval_strategies == {}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestPrecomputedEdgeCases:
    """Edge cases and graceful degradation."""

    def test_no_product_id_skips_precomputed(self):
        """If source has no product_id, precomputed lookup should be skipped."""
        engine, _ = _make_engine()

        # Source with no product_id
        source = _make_profile(
            product_id=None,
            broad_category="tops",
            gemini_category_l1="Tops",
        )

        with patch.object(engine, "_fetch_precomputed_pool") as mock_fetch, \
             patch.object(engine, "_encode_texts_batch", side_effect=Exception("skip")), \
             patch.object(engine, "_retrieve_candidates", return_value=[]), \
             patch.object(engine, "_retrieve_text_candidates", return_value=[]), \
             patch("config.settings.get_settings") as mock_settings:
            mock_settings.return_value = SimpleNamespace(
                use_precomputed_pools=True,
                use_styling_scorer=False,
                use_llm_judge=False,
            )

            engine._score_category(source, "tops", "bottoms", "casual")

        mock_fetch.assert_not_called()

    def test_empty_product_id_skips_precomputed(self):
        """If source has empty string product_id, precomputed lookup should be skipped."""
        engine, _ = _make_engine()

        source = _make_profile(
            product_id="",
            broad_category="tops",
            gemini_category_l1="Tops",
        )

        with patch.object(engine, "_fetch_precomputed_pool") as mock_fetch, \
             patch.object(engine, "_encode_texts_batch", side_effect=Exception("skip")), \
             patch.object(engine, "_retrieve_candidates", return_value=[]), \
             patch.object(engine, "_retrieve_text_candidates", return_value=[]), \
             patch("config.settings.get_settings") as mock_settings:
            mock_settings.return_value = SimpleNamespace(
                use_precomputed_pools=True,
                use_styling_scorer=False,
                use_llm_judge=False,
            )

            engine._score_category(source, "tops", "bottoms", "casual")

        mock_fetch.assert_not_called()

    def test_settings_import_failure_defaults_to_enabled(self):
        """If settings import fails, precomputed pools should still be tried."""
        engine, _ = _make_engine()
        source = _source_top()
        candidates = []
        for i in range(3):
            p = _make_profile(
                product_id=f"cand-{i}",
                broad_category="bottoms",
                gemini_category_l1="Bottoms",
                gemini_category_l2="Jeans",
                formality="Smart Casual",
                fit_type="Regular",
                pattern="Solid",
                apparent_fabric="Denim",
                silhouette="Straight",
                price=50,
            )
            p.similarity = 0.85
            candidates.append(p)

        with patch.object(engine, "_fetch_precomputed_pool", return_value=candidates) as mock_fetch, \
             patch("config.settings.get_settings", side_effect=Exception("no settings")):

            result = engine._score_category(
                source, "tops", "bottoms", "casual",
            )

        # Should still have called precomputed (default=True on import failure)
        mock_fetch.assert_called_once()
        assert len(result) > 0

    def test_fetch_pool_handles_missing_fields_gracefully(self):
        """RPC rows with missing optional fields should still produce valid profiles."""
        engine, mock_sb = _make_engine()

        # Minimal row — many gemini fields missing
        row = {
            "candidate_id": "cand-minimal",
            "cosine_similarity": 0.75,
            "rank": 1,
            "name": "Basic Top",
            "brand": "NoBrand",
            "category": "Tops",
            "broad_category": "tops",
            "price": 30.0,
            "primary_image_url": "https://img/min.jpg",
            "gallery_images": None,
            "colors": None,
            "materials": None,
            "gemini_category_l1": "Tops",
            "gemini_category_l2": None,
            "gemini_occasions": None,
            "gemini_style_tags": None,
            "gemini_pattern": None,
            "gemini_formality": None,
            "gemini_fit_type": None,
            "gemini_color_family": None,
            "gemini_primary_color": None,
            "gemini_secondary_colors": None,
            "gemini_seasons": None,
            "gemini_silhouette": None,
            "gemini_construction": None,
            "gemini_apparent_fabric": None,
            "gemini_texture": None,
            "gemini_coverage_level": None,
            "gemini_sheen": None,
            "gemini_rise": None,
            "gemini_leg_shape": None,
            "gemini_stretch": None,
            "gemini_styling_metadata": None,
            "gemini_styling_role": None,
            "gemini_appearance_top_tags": None,
            "gemini_vibe_tags": None,
            "gemini_extractor_version": None,
        }
        mock_sb.rpc.return_value.execute.return_value = SimpleNamespace(data=[row])

        result = engine._fetch_precomputed_pool("src-001", "tops")

        assert result is not None
        assert len(result) == 1
        p = result[0]
        assert p.product_id == "cand-minimal"
        assert p.gemini_category_l1 == "Tops"
        assert abs(p.similarity - 0.75) < 0.001
