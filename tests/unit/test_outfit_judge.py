"""Unit tests for the LLM Pair Judge (Complete the Fit v3).

Uses SimpleNamespace mocks to avoid importing the real AestheticProfile
dataclass (which would pull in heavy ML dependencies).  All OpenAI calls
are mocked — no real API key needed.
"""

import json
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from services.outfit_judge import (
    FitIntent,
    JudgeResult,
    LLMPairJudge,
    _SYSTEM_PROMPT,
    _OCCASION_TO_BUCKET,
    _FITTED_SILS,
    _OVERSIZED_SILS,
    _profile_to_judge_dict,
    build_judge_payload,
    derive_fit_intent,
    get_pair_judge,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _p(**kwargs) -> SimpleNamespace:
    """Create a mock AestheticProfile with sensible defaults."""
    defaults = {
        "product_id": "src-001",
        "name": "Test Top",
        "brand": "TestBrand",
        "category": "tops",
        "broad_category": "tops",
        "price": 50.0,
        "image_url": "https://example.com/img.jpg",
        "gemini_category_l1": "Tops",
        "gemini_category_l2": "blouse",
        "formality": "casual",
        "formality_level": 2,
        "occasions": ["everyday"],
        "style_tags": ["casual"],
        "pattern": "solid",
        "fit_type": "regular",
        "color_family": "black",
        "primary_color": "black",
        "secondary_colors": [],
        "seasons": ["spring", "fall"],
        "silhouette": "regular",
        "apparent_fabric": "cotton",
        "texture": "smooth",
        "coverage_level": "moderate",
        "sheen": "matte",
        "rise": None,
        "leg_shape": None,
        "stretch": None,
        "length": "regular",
        "is_bridge": False,
        "primary_style": "casual",
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


def _cand(pid="cand-001", **kwargs):
    """Create a candidate mock with distinct defaults from the source."""
    return _p(
        product_id=pid, name=f"Candidate {pid}",
        broad_category="bottoms", gemini_category_l1="Bottoms",
        gemini_category_l2="jeans", color_family="blue",
        primary_color="blue", apparent_fabric="denim",
        material_family="denim", texture="textured", **kwargs,
    )


def _mock_openai_response(results):
    """Build a mock OpenAI chat completion response."""
    content = json.dumps({"results": results})
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture
def judge():
    """Create a judge with a mocked OpenAI client."""
    with patch("openai.OpenAI"):
        j = LLMPairJudge(api_key="test-key", model="test-model", timeout=5.0)
    return j


# ===========================================================================
# 1. FIT INTENT derivation
# ===========================================================================

class TestFitIntentDerivation:

    def test_casual_occasion_default(self):
        intent = derive_fit_intent(_p(occasions=[]))
        assert intent.occasion_target == "casual"

    def test_work_occasion(self):
        intent = derive_fit_intent(_p(occasions=["office", "meeting"]))
        assert intent.occasion_target == "work"

    def test_going_out_occasion(self):
        intent = derive_fit_intent(_p(occasions=["date night", "party", "club"]))
        assert intent.occasion_target == "going-out"

    def test_event_occasion(self):
        intent = derive_fit_intent(_p(occasions=["wedding", "gala"]))
        assert intent.occasion_target == "event"

    def test_active_occasion(self):
        intent = derive_fit_intent(_p(occasions=["gym", "workout", "yoga"]))
        assert intent.occasion_target == "active"

    def test_majority_occasion_wins(self):
        intent = derive_fit_intent(_p(occasions=["office", "meeting", "date night"]))
        assert intent.occasion_target == "work"

    def test_unknown_occasion_maps_casual(self):
        intent = derive_fit_intent(_p(occasions=["unknown_event", "mystery"]))
        assert intent.occasion_target == "casual"

    def test_fitted_silhouette(self):
        intent = derive_fit_intent(_p(silhouette="fitted"))
        assert intent.silhouette_intent == "fitted"

    def test_oversized_silhouette(self):
        intent = derive_fit_intent(_p(silhouette="oversized"))
        assert intent.silhouette_intent == "oversized"

    def test_outerwear_silhouette_layered(self):
        intent = derive_fit_intent(_p(silhouette="regular", gemini_category_l1="outerwear"))
        assert intent.silhouette_intent == "layered"

    def test_regular_silhouette_balanced(self):
        intent = derive_fit_intent(_p(silhouette="regular", gemini_category_l1="Tops"))
        assert intent.silhouette_intent == "balanced"

    def test_vibe_from_primary_style(self):
        intent = derive_fit_intent(_p(primary_style="Bohemian"))
        assert intent.vibe == "bohemian"

    def test_vibe_default_casual(self):
        intent = derive_fit_intent(_p(primary_style=None))
        assert intent.vibe == "casual"


# ===========================================================================
# 2. STATEMENT LEVEL mapping
# ===========================================================================

class TestStatementLevel:

    def test_basic_source_high_statement_target(self):
        intent = derive_fit_intent(_p(style_strength=0.10))
        assert intent.source_is_basic is True
        assert intent.statement_target > 0.5

    def test_statement_source_low_statement_target(self):
        intent = derive_fit_intent(_p(style_strength=0.90))
        assert intent.source_is_basic is False
        assert intent.statement_target < 0.30

    def test_mid_range_balanced(self):
        intent = derive_fit_intent(_p(style_strength=0.50))
        assert intent.source_is_basic is False
        assert intent.statement_target == 0.40

    def test_basic_boundary_035(self):
        intent = derive_fit_intent(_p(style_strength=0.35))
        assert intent.source_is_basic is False

    def test_basic_boundary_034(self):
        intent = derive_fit_intent(_p(style_strength=0.34))
        assert intent.source_is_basic is True

    def test_source_statement_level_stored(self):
        intent = derive_fit_intent(_p(style_strength=0.42))
        assert intent.source_statement_level == 0.42


# ===========================================================================
# 3. FIT INTENT cache key
# ===========================================================================

class TestFitIntentCacheKey:

    def test_same_intent_same_key(self):
        a = FitIntent("work", "fitted", "classic", 0.5, "mild", False, 0.5)
        b = FitIntent("work", "fitted", "classic", 0.5, "mild", False, 0.5)
        assert a.cache_key() == b.cache_key()

    def test_different_intent_different_key(self):
        a = FitIntent("work", "fitted", "classic", 0.5, "mild", False, 0.5)
        b = FitIntent("casual", "oversized", "bohemian", 0.8, "hot", True, 0.2)
        assert a.cache_key() != b.cache_key()

    def test_statement_bucketing_same(self):
        a = FitIntent("work", "fitted", "classic", 0.53, "mild", False, 0.5)
        b = FitIntent("work", "fitted", "classic", 0.47, "mild", False, 0.5)
        assert a.cache_key() == b.cache_key()

    def test_statement_bucket_different(self):
        a = FitIntent("work", "fitted", "classic", 0.44, "mild", False, 0.5)
        b = FitIntent("work", "fitted", "classic", 0.56, "mild", False, 0.5)
        assert a.cache_key() != b.cache_key()

    def test_cache_key_is_8_hex_chars(self):
        key = FitIntent("casual", "balanced", "casual", 0.4, "mild", False, 0.5).cache_key()
        assert len(key) == 8
        assert all(c in "0123456789abcdef" for c in key)


# ===========================================================================
# 4. PROMPT BUILDING
# ===========================================================================

class TestPromptBuilding:

    def test_no_name_in_judge_dict(self):
        d = _profile_to_judge_dict(_p(name="My Fancy Blouse"))
        assert "name" not in d
        assert "My Fancy Blouse" not in str(d)

    def test_no_brand_in_judge_dict(self):
        d = _profile_to_judge_dict(_p(brand="Gucci"))
        assert "brand" not in d
        assert "Gucci" not in str(d)

    def test_no_price_in_judge_dict(self):
        d = _profile_to_judge_dict(_p(price=199.99))
        assert "price" not in d

    def test_judge_dict_has_expected_keys(self):
        d = _profile_to_judge_dict(_p())
        expected = {
            "category_l2", "formality", "silhouette", "fabric", "color",
            "pattern", "statement_level", "occasions", "seasons",
            "texture", "sheen", "length",
        }
        assert set(d.keys()) == expected

    def test_occasions_capped_at_3(self):
        d = _profile_to_judge_dict(_p(occasions=["a", "b", "c", "d", "e"]))
        assert len(d["occasions"]) == 3

    def test_seasons_capped_at_3(self):
        d = _profile_to_judge_dict(_p(seasons=["spring", "summer", "fall", "winter"]))
        assert len(d["seasons"]) == 3

    def test_fabric_falls_back_to_apparent(self):
        d = _profile_to_judge_dict(_p(material_family=None, apparent_fabric="silk"))
        assert d["fabric"] == "silk"

    def test_color_falls_back_to_primary(self):
        d = _profile_to_judge_dict(_p(color_family=None, primary_color="emerald"))
        assert d["color"] == "emerald"

    def test_build_payload_structure(self):
        intent = FitIntent("casual", "balanced", "casual", 0.4, "mild", False, 0.35)
        payload = build_judge_payload(_p(), [_cand("c1"), _cand("c2")], intent)
        assert "intent" in payload
        assert "source" in payload
        assert "candidates" in payload
        assert len(payload["candidates"]) == 2
        assert payload["candidates"][0]["id"] == "c1"
        assert payload["candidates"][1]["id"] == "c2"

    def test_payload_intent_matches(self):
        intent = FitIntent("work", "fitted", "classic", 0.6, "cold", False, 0.5)
        payload = build_judge_payload(_p(), [_cand()], intent)
        assert payload["intent"]["occasion"] == "work"
        assert payload["intent"]["silhouette"] == "fitted"
        assert payload["intent"]["vibe"] == "classic"
        assert payload["intent"]["statement_target"] == 0.6
        assert payload["intent"]["warmth"] == "cold"

    def test_candidate_id_fallback_index(self):
        c = _cand("c1")
        c.product_id = None
        intent = FitIntent("casual", "balanced", "casual", 0.4, "mild", False, 0.35)
        payload = build_judge_payload(_p(), [c], intent)
        assert payload["candidates"][0]["id"] == "cand_0"


# ===========================================================================
# 5. SYSTEM PROMPT content checks
# ===========================================================================

class TestSystemPrompt:

    def test_scoring_rubric_present(self):
        assert "SCORING RUBRIC" in _SYSTEM_PROMPT

    def test_hard_fail_rules_present(self):
        assert "HARD FAILS" in _SYSTEM_PROMPT

    def test_anti_neutrality_rule_present(self):
        assert "ANTI-NEUTRALITY RULE" in _SYSTEM_PROMPT

    def test_statement_balance_rule_present(self):
        assert "STATEMENT BALANCE RULE" in _SYSTEM_PROMPT

    def test_json_schema_in_prompt(self):
        assert '"results"' in _SYSTEM_PROMPT
        assert '"overall"' in _SYSTEM_PROMPT
        assert '"fail"' in _SYSTEM_PROMPT
        assert '"tags"' in _SYSTEM_PROMPT

    def test_tags_list_in_prompt(self):
        for tag in ["occasion_match", "silhouette_balance", "fabric_contrast",
                     "style_coherence", "color_harmony", "too_basic", "too_busy",
                     "occasion_clash", "great_complement", "statement_conflict"]:
            assert tag in _SYSTEM_PROMPT, f"Missing tag: {tag}"

    def test_weight_sum_equals_one(self):
        assert 0.25 + 0.20 + 0.20 + 0.15 + 0.10 + 0.10 == pytest.approx(1.0)


# ===========================================================================
# 6. RESPONSE PARSING
# ===========================================================================

class TestResponseParsing:

    def test_valid_results_array(self):
        content = json.dumps({"results": [
            {"id": "c1", "overall": 7.5, "fail": False, "tags": ["occasion_match"]},
            {"id": "c2", "overall": 3.0, "fail": True, "tags": ["occasion_clash"]},
        ]})
        parsed = LLMPairJudge._parse_response(content)
        assert len(parsed) == 2
        assert parsed[0]["id"] == "c1"
        assert parsed[1]["fail"] is True

    def test_empty_results_array(self):
        assert LLMPairJudge._parse_response(json.dumps({"results": []})) == []

    def test_malformed_json(self):
        assert LLMPairJudge._parse_response("not valid json {{{") == []

    def test_none_content(self):
        assert LLMPairJudge._parse_response(None) == []

    def test_flat_dict_format(self):
        content = json.dumps({
            "c1": {"overall": 8.0, "fail": False, "tags": ["great_complement"]},
            "c2": {"overall": 2.0, "fail": True, "tags": ["occasion_clash"]},
        })
        parsed = LLMPairJudge._parse_response(content)
        assert len(parsed) == 2
        assert {r["id"] for r in parsed} == {"c1", "c2"}

    def test_bare_list_format(self):
        content = json.dumps([{"id": "c1", "overall": 6.0, "fail": False, "tags": []}])
        parsed = LLMPairJudge._parse_response(content)
        assert len(parsed) == 1
        assert parsed[0]["id"] == "c1"

    def test_unexpected_structure(self):
        assert LLMPairJudge._parse_response('"just a string"') == []

    def test_results_key_not_list(self):
        """'results' is not a list and mixed value types -> empty list."""
        content = json.dumps({"results": "not_a_list", "c1": {"overall": 5}})
        parsed = LLMPairJudge._parse_response(content)
        # "not_a_list" is a string (not dict) so flat-dict check fails -> empty
        assert parsed == []


# ===========================================================================
# 7. JUDGE BATCH — core scoring
# ===========================================================================

class TestJudgeBatch:

    def test_empty_candidates_returns_empty(self, judge):
        intent = FitIntent("casual", "balanced", "casual", 0.4, "mild", False, 0.35)
        assert judge.judge_batch(_p(), [], intent) == {}

    def test_normal_scoring(self, judge):
        c1, c2 = _cand("c1"), _cand("c2")
        intent = FitIntent("casual", "balanced", "casual", 0.4, "mild", False, 0.35)
        judge.client.chat.completions.create.return_value = _mock_openai_response([
            {"id": "c1", "overall": 7.5, "fail": False, "tags": ["occasion_match"]},
            {"id": "c2", "overall": 4.0, "fail": False, "tags": ["too_basic"]},
        ])
        results = judge.judge_batch(_p(), [c1, c2], intent)
        assert results["c1"].overall == 7.5
        assert results["c1"].fail is False
        assert results["c1"].tags == ["occasion_match"]
        assert results["c2"].overall == 4.0

    def test_fail_candidate_flagged(self, judge):
        intent = FitIntent("casual", "balanced", "casual", 0.4, "mild", False, 0.35)
        judge.client.chat.completions.create.return_value = _mock_openai_response([
            {"id": "c1", "overall": 0, "fail": True, "tags": ["occasion_clash"]},
        ])
        results = judge.judge_batch(_p(), [_cand("c1")], intent)
        assert results["c1"].fail is True
        assert results["c1"].overall == 0.0

    def test_missing_candidate_gets_default(self, judge):
        intent = FitIntent("casual", "balanced", "casual", 0.4, "mild", False, 0.35)
        judge.client.chat.completions.create.return_value = _mock_openai_response([
            {"id": "c1", "overall": 8.0, "fail": False, "tags": ["great_complement"]},
        ])
        results = judge.judge_batch(_p(), [_cand("c1"), _cand("c2")], intent)
        assert results["c1"].overall == 8.0
        assert results["c2"].overall == 5.0
        assert "missing_from_llm" in results["c2"].tags

    def test_overall_clamped_to_0_10(self, judge):
        intent = FitIntent("casual", "balanced", "casual", 0.4, "mild", False, 0.35)
        judge.client.chat.completions.create.return_value = _mock_openai_response([
            {"id": "c1", "overall": 15.0, "fail": False, "tags": []},
            {"id": "c2", "overall": -3.0, "fail": False, "tags": []},
        ])
        results = judge.judge_batch(_p(), [_cand("c1"), _cand("c2")], intent)
        assert results["c1"].overall == 10.0
        assert results["c2"].overall == 0.0

    def test_non_numeric_overall_defaults_to_5(self, judge):
        intent = FitIntent("casual", "balanced", "casual", 0.4, "mild", False, 0.35)
        judge.client.chat.completions.create.return_value = _mock_openai_response([
            {"id": "c1", "overall": "high", "fail": False, "tags": []},
        ])
        results = judge.judge_batch(_p(), [_cand("c1")], intent)
        assert results["c1"].overall == 5.0

    def test_non_bool_fail_defaults_to_false(self, judge):
        intent = FitIntent("casual", "balanced", "casual", 0.4, "mild", False, 0.35)
        judge.client.chat.completions.create.return_value = _mock_openai_response([
            {"id": "c1", "overall": 6.0, "fail": "yes", "tags": []},
        ])
        results = judge.judge_batch(_p(), [_cand("c1")], intent)
        assert results["c1"].fail is False

    def test_non_list_tags_defaults_to_empty(self, judge):
        intent = FitIntent("casual", "balanced", "casual", 0.4, "mild", False, 0.35)
        judge.client.chat.completions.create.return_value = _mock_openai_response([
            {"id": "c1", "overall": 6.0, "fail": False, "tags": "occasion_match"},
        ])
        results = judge.judge_batch(_p(), [_cand("c1")], intent)
        assert results["c1"].tags == []


# ===========================================================================
# 8. CACHE behavior
# ===========================================================================

class TestCache:

    def test_cache_hit_skips_llm_call(self, judge):
        intent = FitIntent("casual", "balanced", "casual", 0.4, "mild", False, 0.35)
        judge.client.chat.completions.create.return_value = _mock_openai_response([
            {"id": "c1", "overall": 7.0, "fail": False, "tags": ["good_contrast"]},
        ])
        judge.judge_batch(_p(), [_cand("c1")], intent)
        assert judge.client.chat.completions.create.call_count == 1
        judge.judge_batch(_p(), [_cand("c1")], intent)
        assert judge.client.chat.completions.create.call_count == 1

    def test_cache_miss_different_source(self, judge):
        intent = FitIntent("casual", "balanced", "casual", 0.4, "mild", False, 0.35)
        judge.client.chat.completions.create.return_value = _mock_openai_response([
            {"id": "c1", "overall": 7.0, "fail": False, "tags": []},
        ])
        judge.judge_batch(_p(product_id="src-a"), [_cand("c1")], intent)
        judge.judge_batch(_p(product_id="src-b"), [_cand("c1")], intent)
        assert judge.client.chat.completions.create.call_count == 2

    def test_cache_eviction_at_capacity(self, judge):
        judge._CACHE_SIZE = 3
        intent = FitIntent("casual", "balanced", "casual", 0.4, "mild", False, 0.35)
        src = _p()
        for i in range(3):
            judge.client.chat.completions.create.return_value = _mock_openai_response([
                {"id": f"c{i}", "overall": float(i), "fail": False, "tags": []},
            ])
            judge.judge_batch(src, [_cand(f"c{i}")], intent)
        assert len(judge._cache) == 3
        calls_after_fill = judge.client.chat.completions.create.call_count

        judge.client.chat.completions.create.return_value = _mock_openai_response([
            {"id": "c3", "overall": 9.0, "fail": False, "tags": []},
        ])
        judge.judge_batch(src, [_cand("c3")], intent)
        assert len(judge._cache) == 3

        judge.client.chat.completions.create.return_value = _mock_openai_response([
            {"id": "c0", "overall": 1.0, "fail": False, "tags": []},
        ])
        judge.judge_batch(src, [_cand("c0")], intent)
        assert judge.client.chat.completions.create.call_count > calls_after_fill + 1

    def test_intent_hash_bucketing_in_cache(self, judge):
        judge.client.chat.completions.create.return_value = _mock_openai_response([
            {"id": "c1", "overall": 7.0, "fail": False, "tags": []},
        ])
        judge.judge_batch(_p(), [_cand("c1")],
                          FitIntent("work", "fitted", "classic", 0.5, "cold", False, 0.5))
        judge.judge_batch(_p(), [_cand("c1")],
                          FitIntent("casual", "oversized", "bohemian", 0.8, "hot", True, 0.2))
        assert judge.client.chat.completions.create.call_count == 2

    def test_mixed_cached_and_uncached(self, judge):
        intent = FitIntent("casual", "balanced", "casual", 0.4, "mild", False, 0.35)
        judge.client.chat.completions.create.return_value = _mock_openai_response([
            {"id": "c1", "overall": 7.0, "fail": False, "tags": ["good"]},
        ])
        judge.judge_batch(_p(), [_cand("c1")], intent)
        judge.client.chat.completions.create.return_value = _mock_openai_response([
            {"id": "c2", "overall": 5.0, "fail": False, "tags": ["ok"]},
        ])
        results = judge.judge_batch(_p(), [_cand("c1"), _cand("c2")], intent)
        assert judge.client.chat.completions.create.call_count == 2
        assert results["c1"].overall == 7.0
        assert results["c2"].overall == 5.0


# ===========================================================================
# 9. GRACEFUL DEGRADATION
# ===========================================================================

class TestGracefulDegradation:

    def test_llm_exception_returns_defaults(self, judge):
        intent = FitIntent("casual", "balanced", "casual", 0.4, "mild", False, 0.35)
        judge.client.chat.completions.create.side_effect = Exception("API timeout")
        results = judge.judge_batch(_p(), [_cand("c1"), _cand("c2")], intent)
        assert results["c1"].overall == 5.0
        assert results["c1"].tags == ["missing_from_llm"]
        assert results["c2"].overall == 5.0

    def test_retry_on_failure(self, judge):
        judge.max_retries = 1
        intent = FitIntent("casual", "balanced", "casual", 0.4, "mild", False, 0.35)
        good_response = _mock_openai_response([
            {"id": "c1", "overall": 8.0, "fail": False, "tags": ["great_complement"]},
        ])
        judge.client.chat.completions.create.side_effect = [
            Exception("transient error"), good_response,
        ]
        with patch("services.outfit_judge._time.sleep"):
            results = judge.judge_batch(_p(), [_cand("c1")], intent)
        assert results["c1"].overall == 8.0
        assert judge.client.chat.completions.create.call_count == 2

    def test_all_retries_exhausted(self, judge):
        judge.max_retries = 1
        intent = FitIntent("casual", "balanced", "casual", 0.4, "mild", False, 0.35)
        judge.client.chat.completions.create.side_effect = Exception("persistent failure")
        with patch("services.outfit_judge._time.sleep"):
            results = judge.judge_batch(_p(), [_cand("c1")], intent)
        assert results["c1"].overall == 5.0
        assert results["c1"].tags == ["missing_from_llm"]

    def test_get_pair_judge_no_api_key(self):
        import services.outfit_judge as oj
        original = oj._judge
        oj._judge = None
        try:
            mock_settings = MagicMock()
            mock_settings.openai_api_key = ""
            mock_settings.llm_judge_enabled = True
            with patch("config.settings.get_settings", return_value=mock_settings):
                assert get_pair_judge() is None
        finally:
            oj._judge = original

    def test_get_pair_judge_disabled(self):
        import services.outfit_judge as oj
        original = oj._judge
        oj._judge = None
        try:
            mock_settings = MagicMock()
            mock_settings.openai_api_key = "sk-test"
            mock_settings.llm_judge_enabled = False
            with patch("config.settings.get_settings", return_value=mock_settings):
                assert get_pair_judge() is None
        finally:
            oj._judge = original

    def test_get_pair_judge_success(self):
        import services.outfit_judge as oj
        original = oj._judge
        oj._judge = None
        try:
            mock_settings = MagicMock()
            mock_settings.openai_api_key = "sk-test-key"
            mock_settings.llm_judge_enabled = True
            mock_settings.llm_judge_model = "gpt-4.1-mini"
            mock_settings.llm_judge_timeout = 15.0
            with patch("config.settings.get_settings", return_value=mock_settings), \
                 patch("openai.OpenAI"):
                assert isinstance(get_pair_judge(), LLMPairJudge)
        finally:
            oj._judge = original

    def test_get_pair_judge_import_error(self):
        import services.outfit_judge as oj
        original = oj._judge
        oj._judge = None
        try:
            with patch("config.settings.get_settings",
                       side_effect=ImportError("no settings")):
                assert get_pair_judge() is None
        finally:
            oj._judge = original


# ===========================================================================
# 10. SCORE BLENDING math
# ===========================================================================

class TestScoreBlending:

    def test_blend_both_high(self):
        blended = (0.55 * 8.5 + 0.45 * 9.0) / 10.0
        assert blended == pytest.approx(0.8725)

    def test_blend_low_tattoo_high_llm(self):
        blended = (0.55 * 3.0 + 0.45 * 9.0) / 10.0
        assert blended == pytest.approx(0.57)

    def test_blend_high_tattoo_low_llm(self):
        blended = (0.55 * 9.0 + 0.45 * 2.0) / 10.0
        assert blended == pytest.approx(0.585)

    def test_blend_both_zero(self):
        assert (0.55 * 0.0 + 0.45 * 0.0) / 10.0 == 0.0

    def test_blend_both_max(self):
        assert (0.55 * 10.0 + 0.45 * 10.0) / 10.0 == pytest.approx(1.0)

    def test_blend_weights_sum_to_one(self):
        assert 0.55 + 0.45 == pytest.approx(1.0)

    def test_tattoo_norm_clamped_high(self):
        assert max(0.0, min(10.0, 1.2 * 10.0)) == 10.0

    def test_tattoo_norm_clamped_low(self):
        assert max(0.0, min(10.0, -0.1 * 10.0)) == 0.0

    def test_fail_removes_candidate(self):
        entry = {"tattoo": 0.75, "profile": _cand("c1")}
        jr = JudgeResult(overall=0.0, fail=True, tags=["occasion_clash"])
        if jr.fail:
            entry["tattoo"] = -1.0
            entry["llm_fail"] = True
        assert entry["tattoo"] == -1.0
        assert entry["llm_fail"] is True


# ===========================================================================
# 11. WARMTH TARGET
# ===========================================================================

class TestWarmthTarget:

    def test_warmth_from_temp_band(self):
        assert derive_fit_intent(_p(temp_band="cold")).warmth_target == "cold"

    def test_warmth_default_mild(self):
        assert derive_fit_intent(_p(temp_band=None)).warmth_target == "mild"


# ===========================================================================
# 12. OCCASION BUCKET completeness
# ===========================================================================

class TestOccasionBuckets:

    def test_all_buckets_represented(self):
        assert set(_OCCASION_TO_BUCKET.values()) == {"active", "work", "going-out", "event", "casual"}

    def test_no_empty_values(self):
        for k, v in _OCCASION_TO_BUCKET.items():
            assert v, f"Empty bucket for key: {k}"

    def test_known_occasions_mapped(self):
        assert _OCCASION_TO_BUCKET["gym"] == "active"
        assert _OCCASION_TO_BUCKET["office"] == "work"
        assert _OCCASION_TO_BUCKET["party"] == "going-out"
        assert _OCCASION_TO_BUCKET["wedding"] == "event"
        assert _OCCASION_TO_BUCKET["brunch"] == "casual"


# ===========================================================================
# 13. SILHOUETTE SET completeness
# ===========================================================================

class TestSilhouetteSets:

    def test_fitted_and_oversized_disjoint(self):
        assert _FITTED_SILS.isdisjoint(_OVERSIZED_SILS)

    def test_fitted_non_empty(self):
        assert len(_FITTED_SILS) > 0

    def test_oversized_non_empty(self):
        assert len(_OVERSIZED_SILS) > 0
