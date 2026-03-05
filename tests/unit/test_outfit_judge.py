"""Unit tests for the Stylist Ranking Judge (Complete the Fit v3.3).

Uses SimpleNamespace mocks — no real OpenAI API key needed.
All LLM API calls are mocked.
"""

import json
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from services.outfit_judge import (
    FitIntent,
    StylistJudge,
    VisionJudge,
    _RERANK_SYSTEM_PROMPT,
    _OUTFIT_RANKING_SYSTEM_PROMPT,
    _OCCASION_TO_BUCKET,
    _FITTED_SILS,
    _OVERSIZED_SILS,
    _AVIF_FTYP,
    _AVIF_BRANDS,
    _is_valid_image_url,
    _is_supported_image_format,
    build_rerank_messages,
    build_vision_messages,
    build_outfit_ranking_messages,
    derive_fit_intent,
    get_pair_judge,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _p(**kwargs) -> SimpleNamespace:
    defaults = {
        "product_id": "src-001", "name": "Test Top", "brand": "TestBrand",
        "category": "tops", "broad_category": "tops", "price": 50.0,
        "image_url": "https://cdn.example.com/src.jpg",
        "gemini_category_l1": "Tops", "gemini_category_l2": "blouse",
        "formality": "casual", "formality_level": 2,
        "occasions": ["everyday"], "style_tags": ["casual"],
        "pattern": "solid", "fit_type": "regular",
        "color_family": "black", "primary_color": "black",
        "secondary_colors": [], "seasons": ["spring", "fall"],
        "silhouette": "regular", "apparent_fabric": "cotton",
        "texture": "smooth", "coverage_level": "moderate",
        "sheen": "matte", "rise": None, "leg_shape": None,
        "stretch": None, "length": "regular", "is_bridge": False,
        "primary_style": "casual", "style_strength": 0.35,
        "material_family": "cotton", "texture_intensity": "smooth",
        "shine_level": "matte", "fabric_weight": "mid",
        "layer_role": "base", "temp_band": "mild",
        "color_saturation": "medium", "similarity": 0.5,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _cand(pid="cand-001", **kwargs):
    defaults = dict(
        product_id=pid, name=f"Candidate {pid}",
        image_url=f"https://cdn.example.com/{pid}.jpg",
        broad_category="bottoms", gemini_category_l1="Bottoms",
        gemini_category_l2="jeans", color_family="blue",
        primary_color="blue", apparent_fabric="denim",
        material_family="denim", texture="textured",
    )
    defaults.update(kwargs)
    return _p(**defaults)


def _mock_ranking_response(ranking, note="test"):
    """Build a mock OpenAI response with a ranking list."""
    content = json.dumps({"ranking": ranking, "note": note})
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture
def judge():
    with patch("openai.OpenAI"):
        j = StylistJudge(api_key="test-key", model="gpt-4.1", timeout=5.0)
    return j


# ===========================================================================
# 1. FIT INTENT derivation
# ===========================================================================

class TestFitIntentDerivation:

    def test_casual_occasion_default(self):
        assert derive_fit_intent(_p(occasions=[])).occasion_target == "casual"

    def test_work_occasion(self):
        assert derive_fit_intent(_p(occasions=["office", "meeting"])).occasion_target == "work"

    def test_going_out_occasion(self):
        assert derive_fit_intent(_p(occasions=["party", "club"])).occasion_target == "going-out"

    def test_event_occasion(self):
        assert derive_fit_intent(_p(occasions=["wedding", "gala"])).occasion_target == "event"

    def test_active_occasion(self):
        assert derive_fit_intent(_p(occasions=["gym", "workout"])).occasion_target == "active"

    def test_majority_occasion_wins(self):
        assert derive_fit_intent(_p(occasions=["office", "meeting", "party"])).occasion_target == "work"

    def test_unknown_occasion_maps_casual(self):
        assert derive_fit_intent(_p(occasions=["unknown_thing"])).occasion_target == "casual"

    def test_fitted_silhouette(self):
        assert derive_fit_intent(_p(silhouette="fitted")).silhouette_intent == "fitted"

    def test_oversized_silhouette(self):
        assert derive_fit_intent(_p(silhouette="oversized")).silhouette_intent == "oversized"

    def test_outerwear_layered(self):
        assert derive_fit_intent(_p(silhouette="regular", gemini_category_l1="outerwear")).silhouette_intent == "layered"

    def test_regular_balanced(self):
        assert derive_fit_intent(_p(silhouette="regular", gemini_category_l1="Tops")).silhouette_intent == "balanced"

    def test_vibe_from_primary_style(self):
        assert derive_fit_intent(_p(primary_style="Bohemian")).vibe == "bohemian"

    def test_vibe_default_casual(self):
        assert derive_fit_intent(_p(primary_style=None)).vibe == "casual"

    def test_source_category_from_l2(self):
        assert derive_fit_intent(_p(gemini_category_l2="Jogger")).source_category == "Jogger"

    def test_warmth_from_temp_band(self):
        assert derive_fit_intent(_p(temp_band="cold")).warmth_target == "cold"

    def test_warmth_default_mild(self):
        assert derive_fit_intent(_p(temp_band=None)).warmth_target == "mild"


# ===========================================================================
# 2. FIT INTENT cache key + context line
# ===========================================================================

class TestFitIntentCacheKey:

    def test_same_intent_same_key(self):
        a = FitIntent("work", "fitted", "classic", "Blouse", "mild")
        b = FitIntent("work", "fitted", "classic", "Blouse", "mild")
        assert a.cache_key() == b.cache_key()

    def test_different_intent_different_key(self):
        a = FitIntent("work", "fitted", "classic", "Blouse", "mild")
        b = FitIntent("casual", "oversized", "bohemian", "Jogger", "hot")
        assert a.cache_key() != b.cache_key()

    def test_cache_key_is_8_hex(self):
        key = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild").cache_key()
        assert len(key) == 8
        assert all(c in "0123456789abcdef" for c in key)

    def test_context_line_contains_occasion(self):
        intent = FitIntent("work", "fitted", "classic", "Blouse", "mild")
        line = intent.context_line()
        assert "work" in line
        assert "blouse" in line
        assert "classic" in line


# ===========================================================================
# 3. RERANK MESSAGE BUILDING
# ===========================================================================

class TestRerankMessages:

    def test_message_structure(self):
        msgs = build_rerank_messages(_p(), [_cand("c1")], FitIntent("casual", "balanced", "casual", "T-Shirt", "mild"))
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert isinstance(msgs[1]["content"], list)

    def test_system_prompt_is_rerank_prompt(self):
        msgs = build_rerank_messages(_p(), [_cand("c1")], FitIntent("casual", "balanced", "casual", "T-Shirt", "mild"))
        assert msgs[0]["content"] == _RERANK_SYSTEM_PROMPT

    def test_source_image_included(self):
        src = _p(image_url="https://cdn.example.com/source.jpg")
        msgs = build_rerank_messages(src, [_cand("c1")], FitIntent("casual", "balanced", "casual", "T-Shirt", "mild"))
        image_parts = [p for p in msgs[1]["content"] if p.get("type") == "image_url"]
        urls = [p["image_url"]["url"] for p in image_parts]
        assert "https://cdn.example.com/source.jpg" in urls

    def test_candidate_images_included(self):
        c1 = _cand("c1")
        c2 = _cand("c2")
        msgs = build_rerank_messages(_p(), [c1, c2], FitIntent("casual", "balanced", "casual", "T-Shirt", "mild"))
        image_parts = [p for p in msgs[1]["content"] if p.get("type") == "image_url"]
        # 1 source + 2 candidates = 3 images
        assert len(image_parts) == 3

    def test_default_detail_auto(self):
        msgs = build_rerank_messages(_p(), [_cand("c1")], FitIntent("casual", "balanced", "casual", "T-Shirt", "mild"))
        image_parts = [p for p in msgs[1]["content"] if p.get("type") == "image_url"]
        for part in image_parts:
            assert part["image_url"]["detail"] == "auto"

    def test_custom_detail_parameter(self):
        msgs = build_rerank_messages(
            _p(), [_cand("c1")],
            FitIntent("casual", "balanced", "casual", "T-Shirt", "mild"),
            detail="high",
        )
        image_parts = [p for p in msgs[1]["content"] if p.get("type") == "image_url"]
        for part in image_parts:
            assert part["image_url"]["detail"] == "high"

    def test_candidate_id_in_text(self):
        msgs = build_rerank_messages(_p(), [_cand("c1")], FitIntent("casual", "balanced", "casual", "T-Shirt", "mild"))
        text_parts = [p["text"] for p in msgs[1]["content"] if p.get("type") == "text"]
        full_text = " ".join(text_parts)
        assert "c1" in full_text

    def test_context_in_message(self):
        intent = FitIntent("work", "fitted", "classic", "Blouse", "mild")
        msgs = build_rerank_messages(_p(), [_cand("c1")], intent)
        text_parts = [p["text"] for p in msgs[1]["content"] if p.get("type") == "text"]
        full_text = " ".join(text_parts)
        assert "work" in full_text

    def test_ranking_instruction_in_message(self):
        msgs = build_rerank_messages(_p(), [_cand("c1")], FitIntent("casual", "balanced", "casual", "T-Shirt", "mild"))
        text_parts = [p["text"] for p in msgs[1]["content"] if p.get("type") == "text"]
        full_text = " ".join(text_parts)
        assert "BEST" in full_text and "WORST" in full_text

    def test_no_name_or_brand_in_message(self):
        src = _p(name="My Fancy Blouse", brand="Gucci")
        msgs = build_rerank_messages(src, [_cand("c1")], FitIntent("casual", "balanced", "casual", "T-Shirt", "mild"))
        text_parts = [p["text"] for p in msgs[1]["content"] if p.get("type") == "text"]
        full_text = " ".join(text_parts)
        assert "My Fancy Blouse" not in full_text
        assert "Gucci" not in full_text

    def test_no_price_in_message(self):
        src = _p(price=199.99)
        msgs = build_rerank_messages(src, [_cand("c1")], FitIntent("casual", "balanced", "casual", "T-Shirt", "mild"))
        text_parts = [p["text"] for p in msgs[1]["content"] if p.get("type") == "text"]
        full_text = " ".join(text_parts)
        assert "199" not in full_text

    def test_missing_source_image_skips_image_part(self):
        src = _p(image_url="")
        msgs = build_rerank_messages(src, [_cand("c1")], FitIntent("casual", "balanced", "casual", "T-Shirt", "mild"))
        image_parts = [p for p in msgs[1]["content"] if p.get("type") == "image_url"]
        # Only candidate image, no source image
        assert len(image_parts) == 1

    def test_missing_candidate_image_skips_image_part(self):
        c = _cand("c1")
        c.image_url = ""
        msgs = build_rerank_messages(_p(), [c], FitIntent("casual", "balanced", "casual", "T-Shirt", "mild"))
        image_parts = [p for p in msgs[1]["content"] if p.get("type") == "image_url"]
        # Only source image, no candidate image
        assert len(image_parts) == 1

    def test_json_return_instruction(self):
        msgs = build_rerank_messages(_p(), [_cand("c1")], FitIntent("casual", "balanced", "casual", "T-Shirt", "mild"))
        text_parts = [p["text"] for p in msgs[1]["content"] if p.get("type") == "text"]
        full_text = " ".join(text_parts)
        assert "ranking" in full_text.lower()


class TestLegacyVisionMessages:
    """Legacy build_vision_messages delegates to rerank with detail=low."""

    def test_delegates_to_rerank(self):
        msgs = build_vision_messages(_p(), [_cand("c1")], FitIntent("casual", "balanced", "casual", "T-Shirt", "mild"))
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"

    def test_uses_low_detail(self):
        msgs = build_vision_messages(_p(), [_cand("c1")], FitIntent("casual", "balanced", "casual", "T-Shirt", "mild"))
        image_parts = [p for p in msgs[1]["content"] if p.get("type") == "image_url"]
        for part in image_parts:
            assert part["image_url"]["detail"] == "low"


# ===========================================================================
# 4. SYSTEM PROMPTS
# ===========================================================================

class TestStylistPrompts:

    def test_rerank_prompt_has_fashion_criteria(self):
        lower = _RERANK_SYSTEM_PROMPT.lower()
        assert "stylist" in lower
        assert "texture" in lower or "fabric" in lower
        assert "color" in lower

    def test_rerank_prompt_rewards_strong_over_safe(self):
        assert "STRONG" in _RERANK_SYSTEM_PROMPT or "strong" in _RERANK_SYSTEM_PROMPT.lower()
        assert "safe" in _RERANK_SYSTEM_PROMPT.lower()

    def test_rerank_prompt_penalizes_boring(self):
        lower = _RERANK_SYSTEM_PROMPT.lower()
        assert "boring" in lower or "generic" in lower

    def test_rerank_prompt_requests_json(self):
        assert "JSON" in _RERANK_SYSTEM_PROMPT

    def test_outfit_prompt_has_style_story(self):
        lower = _OUTFIT_RANKING_SYSTEM_PROMPT.lower()
        assert "style story" in lower or "story" in lower

    def test_outfit_prompt_rewards_contrast(self):
        lower = _OUTFIT_RANKING_SYSTEM_PROMPT.lower()
        assert "contrast" in lower

    def test_outfit_prompt_penalizes_boring(self):
        lower = _OUTFIT_RANKING_SYSTEM_PROMPT.lower()
        assert "boring" in lower or "generic" in lower

    def test_outfit_prompt_requests_json(self):
        assert "JSON" in _OUTFIT_RANKING_SYSTEM_PROMPT

    def test_rerank_prompt_no_weighted_scoring(self):
        """No weighted scoring dimensions — this is ranking, not rubric."""
        assert "weight 0.25" not in _RERANK_SYSTEM_PROMPT
        assert "SCORING RUBRIC" not in _RERANK_SYSTEM_PROMPT


# ===========================================================================
# 5. RERANK RESPONSE PARSING
# ===========================================================================

class TestRerankResponseParsing:

    def test_valid_ranking_with_ids(self):
        cand_pids = ["c1", "c2", "c3"]
        result = StylistJudge._parse_rerank_response(
            json.dumps({"ranking": ["c2", "c1", "c3"]}), cand_pids,
        )
        assert result == ["c2", "c1", "c3"]

    def test_partial_ranking_fills_missing(self):
        cand_pids = ["c1", "c2", "c3"]
        result = StylistJudge._parse_rerank_response(
            json.dumps({"ranking": ["c3"]}), cand_pids,
        )
        assert result[0] == "c3"
        assert set(result) == {"c1", "c2", "c3"}

    def test_unknown_ids_filtered(self):
        cand_pids = ["c1", "c2"]
        result = StylistJudge._parse_rerank_response(
            json.dumps({"ranking": ["c1", "unknown", "c2"]}), cand_pids,
        )
        assert result == ["c1", "c2"]

    def test_duplicate_ids_filtered(self):
        cand_pids = ["c1", "c2"]
        result = StylistJudge._parse_rerank_response(
            json.dumps({"ranking": ["c1", "c1", "c2"]}), cand_pids,
        )
        assert result == ["c1", "c2"]

    def test_malformed_json(self):
        assert StylistJudge._parse_rerank_response("not json {{{", ["c1"]) is None

    def test_none_content(self):
        assert StylistJudge._parse_rerank_response(None, ["c1"]) is None

    def test_not_a_dict(self):
        assert StylistJudge._parse_rerank_response("[1, 2, 3]", ["c1"]) is None

    def test_missing_ranking_key(self):
        result = StylistJudge._parse_rerank_response(
            json.dumps({"something": "else"}), ["c1"],
        )
        assert result is None

    def test_ranking_not_a_list(self):
        result = StylistJudge._parse_rerank_response(
            json.dumps({"ranking": "c1"}), ["c1"],
        )
        assert result is None

    def test_empty_ranking_list(self):
        result = StylistJudge._parse_rerank_response(
            json.dumps({"ranking": []}), ["c1"],
        )
        assert result is None

    def test_no_matching_ids_returns_none(self):
        result = StylistJudge._parse_rerank_response(
            json.dumps({"ranking": ["x1", "x2"]}), ["c1", "c2"],
        )
        assert result is None

    def test_preserves_tattoo_order_for_unranked(self):
        """Unranked candidates appended in their original order."""
        cand_pids = ["c1", "c2", "c3", "c4"]
        result = StylistJudge._parse_rerank_response(
            json.dumps({"ranking": ["c3", "c1"]}), cand_pids,
        )
        assert result == ["c3", "c1", "c2", "c4"]


# ===========================================================================
# 6. RERANK CATEGORY
# ===========================================================================

class TestRerankCategory:

    def test_empty_candidates_returns_none(self, judge):
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        assert judge.rerank_category(_p(), [], intent) is None

    def test_invalid_source_url_returns_none(self, judge):
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        src = _p(image_url="https://www.example.com/page.html")
        result = judge.rerank_category(src, [_cand("c1")], intent)
        assert result is None

    def test_all_bad_candidate_urls_returns_none(self, judge):
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        bad = _cand("c1", image_url="https://www.example.com/page.html")
        result = judge.rerank_category(_p(), [bad], intent)
        assert result is None

    def test_successful_rerank(self, judge):
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        cands = [_cand("c1"), _cand("c2"), _cand("c3")]
        judge.client.chat.completions.create.return_value = _mock_ranking_response(["c2", "c3", "c1"])
        result = judge.rerank_category(_p(), cands, intent)
        assert result == ["c2", "c3", "c1"]

    def test_partial_rerank_fills_missing(self, judge):
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        cands = [_cand("c1"), _cand("c2"), _cand("c3")]
        judge.client.chat.completions.create.return_value = _mock_ranking_response(["c3"])
        result = judge.rerank_category(_p(), cands, intent)
        assert result[0] == "c3"
        assert set(result) == {"c1", "c2", "c3"}

    def test_filters_bad_candidate_urls(self, judge):
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        good = _cand("c1", image_url="https://cdn.example.com/c1.jpg")
        bad = _cand("c2", image_url="https://www.example.com/page.html")
        judge.client.chat.completions.create.return_value = _mock_ranking_response(["c1"])
        result = judge.rerank_category(_p(), [good, bad], intent)
        # Only c1 was sent to LLM
        assert result is not None
        assert "c1" in result

    def test_uses_json_mode(self, judge):
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        judge.client.chat.completions.create.return_value = _mock_ranking_response(["c1"])
        judge.rerank_category(_p(), [_cand("c1")], intent)
        call_kwargs = judge.client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_uses_temperature_zero(self, judge):
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        judge.client.chat.completions.create.return_value = _mock_ranking_response(["c1"])
        judge.rerank_category(_p(), [_cand("c1")], intent)
        call_kwargs = judge.client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0


# ===========================================================================
# 7. LEGACY judge_batch COMPATIBILITY
# ===========================================================================

class TestLegacyJudgeBatch:

    def test_empty_candidates(self, judge):
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        assert judge.judge_batch(_p(), [], intent) == set()

    def test_converts_ranking_to_bottom_fails(self, judge):
        """Bottom ~30% of ranking become 'fails' for legacy callers."""
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        cands = [_cand(f"c{i}") for i in range(10)]
        # Ranking: c0 best, c9 worst
        ranking = [f"c{i}" for i in range(10)]
        judge.client.chat.completions.create.return_value = _mock_ranking_response(ranking)
        result = judge.judge_batch(_p(), cands, intent)
        # Bottom 3 of 10 should be fails
        assert isinstance(result, set)
        assert len(result) > 0
        # Top-ranked items should NOT be in fails
        assert "c0" not in result
        assert "c1" not in result

    def test_small_set_no_fails(self, judge):
        """3 or fewer candidates → no fails."""
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        cands = [_cand("c1"), _cand("c2")]
        judge.client.chat.completions.create.return_value = _mock_ranking_response(["c1", "c2"])
        result = judge.judge_batch(_p(), cands, intent)
        assert result == set()

    def test_api_failure_returns_empty(self, judge):
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        judge.client.chat.completions.create.side_effect = Exception("timeout")
        result = judge.judge_batch(_p(), [_cand("c1"), _cand("c2")], intent)
        assert result == set()


# ===========================================================================
# 8. CACHE
# ===========================================================================

class TestCache:

    def test_cache_hit_skips_llm(self, judge):
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        judge.client.chat.completions.create.return_value = _mock_ranking_response(["c1"])
        judge.rerank_category(_p(), [_cand("c1")], intent)
        assert judge.client.chat.completions.create.call_count == 1
        # Second call — cached
        judge.rerank_category(_p(), [_cand("c1")], intent)
        assert judge.client.chat.completions.create.call_count == 1

    def test_cached_ranking_remembered(self, judge):
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        cands = [_cand("c1"), _cand("c2")]
        judge.client.chat.completions.create.return_value = _mock_ranking_response(["c2", "c1"])
        judge.rerank_category(_p(), cands, intent)
        result = judge.rerank_category(_p(), cands, intent)
        assert result == ["c2", "c1"]

    def test_cache_miss_different_source(self, judge):
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        judge.client.chat.completions.create.return_value = _mock_ranking_response(["c1"])
        judge.rerank_category(_p(product_id="src-a"), [_cand("c1")], intent)
        judge.rerank_category(_p(product_id="src-b"), [_cand("c1")], intent)
        assert judge.client.chat.completions.create.call_count == 2

    def test_cache_miss_different_candidates(self, judge):
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        judge.client.chat.completions.create.return_value = _mock_ranking_response(["c1"])
        judge.rerank_category(_p(), [_cand("c1")], intent)
        judge.client.chat.completions.create.return_value = _mock_ranking_response(["c2"])
        judge.rerank_category(_p(), [_cand("c2")], intent)
        assert judge.client.chat.completions.create.call_count == 2

    def test_cache_eviction(self, judge):
        judge._CACHE_SIZE = 2
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        judge.client.chat.completions.create.return_value = _mock_ranking_response(["c0"])
        judge.rerank_category(_p(), [_cand("c0")], intent)
        judge.client.chat.completions.create.return_value = _mock_ranking_response(["c1"])
        judge.rerank_category(_p(), [_cand("c1")], intent)
        assert len(judge._cache) == 2
        # Add c2 -> evicts c0
        judge.client.chat.completions.create.return_value = _mock_ranking_response(["c2"])
        judge.rerank_category(_p(), [_cand("c2")], intent)
        assert len(judge._cache) == 2
        # c0 evicted -> needs new call
        calls_before = judge.client.chat.completions.create.call_count
        judge.client.chat.completions.create.return_value = _mock_ranking_response(["c0"])
        judge.rerank_category(_p(), [_cand("c0")], intent)
        assert judge.client.chat.completions.create.call_count > calls_before

    def test_cache_key_deterministic(self):
        k1 = StylistJudge._make_ranking_cache_key("src-1", ["c1", "c2"], "abc")
        k2 = StylistJudge._make_ranking_cache_key("src-1", ["c1", "c2"], "abc")
        assert k1 == k2

    def test_cache_key_order_independent(self):
        """Candidate order shouldn't matter (sorted internally)."""
        k1 = StylistJudge._make_ranking_cache_key("src-1", ["c1", "c2"], "abc")
        k2 = StylistJudge._make_ranking_cache_key("src-1", ["c2", "c1"], "abc")
        assert k1 == k2


# ===========================================================================
# 9. GRACEFUL DEGRADATION
# ===========================================================================

class TestGracefulDegradation:

    def test_exception_returns_none(self, judge):
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        judge.client.chat.completions.create.side_effect = Exception("timeout")
        result = judge.rerank_category(_p(), [_cand("c1"), _cand("c2")], intent)
        assert result is None

    def test_retry_on_failure(self, judge):
        judge.max_retries = 1
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        good = _mock_ranking_response(["c1"])
        judge.client.chat.completions.create.side_effect = [Exception("err"), good]
        with patch("services.outfit_judge._time.sleep"):
            result = judge.rerank_category(_p(), [_cand("c1")], intent)
        assert result == ["c1"]
        assert judge.client.chat.completions.create.call_count == 2

    def test_all_retries_exhausted(self, judge):
        judge.max_retries = 1
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        judge.client.chat.completions.create.side_effect = Exception("persistent")
        with patch("services.outfit_judge._time.sleep"):
            result = judge.rerank_category(_p(), [_cand("c1")], intent)
        assert result is None

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
            mock_settings.llm_judge_model = "gpt-4.1"
            mock_settings.llm_judge_timeout = 30.0
            mock_settings.llm_judge_detail = "auto"
            with patch("config.settings.get_settings", return_value=mock_settings), \
                 patch("openai.OpenAI"):
                result = get_pair_judge()
                assert isinstance(result, StylistJudge)
        finally:
            oj._judge = original

    def test_vision_judge_alias(self):
        assert VisionJudge is StylistJudge


# ===========================================================================
# 10. OCCASION BUCKETS + SILHOUETTE SETS
# ===========================================================================

class TestOccasionBuckets:

    def test_all_buckets_represented(self):
        assert set(_OCCASION_TO_BUCKET.values()) == {"active", "work", "going-out", "event", "casual"}

    def test_known_occasions(self):
        assert _OCCASION_TO_BUCKET["gym"] == "active"
        assert _OCCASION_TO_BUCKET["office"] == "work"
        assert _OCCASION_TO_BUCKET["party"] == "going-out"
        assert _OCCASION_TO_BUCKET["wedding"] == "event"
        assert _OCCASION_TO_BUCKET["brunch"] == "casual"


class TestSilhouetteSets:

    def test_disjoint(self):
        assert _FITTED_SILS.isdisjoint(_OVERSIZED_SILS)

    def test_non_empty(self):
        assert len(_FITTED_SILS) > 0
        assert len(_OVERSIZED_SILS) > 0


# ===========================================================================
# 11. OUTFIT RANKING
# ===========================================================================

def _outfit(cats_and_pids):
    """Build a mock outfit dict: {cat: AestheticProfile, ...}."""
    return {
        cat: _cand(pid=pid, broad_category=cat)
        for cat, pid in cats_and_pids.items()
    }


class TestOutfitRankingMessages:

    def _make_outfits(self):
        return [
            {"bottoms": _cand("b1"), "outerwear": _cand("o1")},
            {"bottoms": _cand("b2"), "outerwear": _cand("o2")},
        ]

    def test_structure(self):
        source = _p()
        outfits = self._make_outfits()
        intent = FitIntent("casual", "balanced", "casual", "blouse", "mild")
        msgs = build_outfit_ranking_messages(source, outfits, intent)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_system_prompt_is_outfit_ranking_prompt(self):
        source = _p()
        outfits = self._make_outfits()
        intent = FitIntent("casual", "balanced", "casual", "blouse", "mild")
        msgs = build_outfit_ranking_messages(source, outfits, intent)
        assert msgs[0]["content"] == _OUTFIT_RANKING_SYSTEM_PROMPT

    def test_source_image_included(self):
        source = _p(image_url="https://cdn.example.com/source.jpg")
        outfits = self._make_outfits()
        intent = FitIntent("casual", "balanced", "casual", "blouse", "mild")
        msgs = build_outfit_ranking_messages(source, outfits, intent)
        images = [
            p for p in msgs[1]["content"]
            if isinstance(p, dict) and p.get("type") == "image_url"
        ]
        source_urls = [img["image_url"]["url"] for img in images]
        assert "https://cdn.example.com/source.jpg" in source_urls

    def test_default_detail_auto(self):
        source = _p()
        outfits = self._make_outfits()
        intent = FitIntent("casual", "balanced", "casual", "blouse", "mild")
        msgs = build_outfit_ranking_messages(source, outfits, intent)
        images = [
            p for p in msgs[1]["content"]
            if isinstance(p, dict) and p.get("type") == "image_url"
        ]
        for img in images:
            assert img["image_url"]["detail"] == "auto"

    def test_outfit_labels_present(self):
        source = _p()
        outfits = self._make_outfits()
        intent = FitIntent("casual", "balanced", "casual", "blouse", "mild")
        msgs = build_outfit_ranking_messages(source, outfits, intent)
        text_parts = [
            p["text"] for p in msgs[1]["content"]
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        full_text = " ".join(text_parts)
        assert "OUTFIT 1" in full_text
        assert "OUTFIT 2" in full_text

    def test_image_count(self):
        """Source + 2 outfits x 2 pieces = 5 images."""
        source = _p()
        outfits = self._make_outfits()
        intent = FitIntent("casual", "balanced", "casual", "blouse", "mild")
        msgs = build_outfit_ranking_messages(source, outfits, intent)
        images = [
            p for p in msgs[1]["content"]
            if isinstance(p, dict) and p.get("type") == "image_url"
        ]
        assert len(images) == 5  # 1 source + 2*2 outfit pieces

    def test_missing_source_image(self):
        source = _p(image_url="")
        outfits = self._make_outfits()
        intent = FitIntent("casual", "balanced", "casual", "blouse", "mild")
        msgs = build_outfit_ranking_messages(source, outfits, intent)
        images = [
            p for p in msgs[1]["content"]
            if isinstance(p, dict) and p.get("type") == "image_url"
        ]
        # Only outfit piece images, no source
        assert len(images) == 4

    def test_context_in_message(self):
        source = _p()
        outfits = self._make_outfits()
        intent = FitIntent("work", "fitted", "minimal", "Shirt", "mild")
        msgs = build_outfit_ranking_messages(source, outfits, intent)
        text_parts = [
            p["text"] for p in msgs[1]["content"]
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        full_text = " ".join(text_parts)
        assert "work" in full_text.lower()


class TestParseOutfitRanking:

    def test_valid_ranking(self):
        result = StylistJudge._parse_outfit_ranking(
            '{"ranking": [3, 1, 2]}', n_outfits=3,
        )
        assert result == [2, 0, 1]  # 0-indexed

    def test_single_winner(self):
        result = StylistJudge._parse_outfit_ranking(
            '{"ranking": [2]}', n_outfits=3,
        )
        # [1] + fill in [0, 2]
        assert result[0] == 1  # winner is outfit index 1

    def test_invalid_json(self):
        result = StylistJudge._parse_outfit_ranking("not json", n_outfits=3)
        assert result is None

    def test_not_a_dict(self):
        result = StylistJudge._parse_outfit_ranking("[1, 2, 3]", n_outfits=3)
        assert result is None

    def test_no_ranking_key(self):
        result = StylistJudge._parse_outfit_ranking('{"result": [1, 2]}', n_outfits=2)
        assert result is None

    def test_ranking_not_a_list(self):
        result = StylistJudge._parse_outfit_ranking('{"ranking": "1,2,3"}', n_outfits=3)
        assert result is None

    def test_out_of_range_filtered(self):
        result = StylistJudge._parse_outfit_ranking(
            '{"ranking": [1, 99, 2]}', n_outfits=2,
        )
        assert result == [0, 1]

    def test_duplicates_filtered(self):
        result = StylistJudge._parse_outfit_ranking(
            '{"ranking": [2, 2, 1]}', n_outfits=2,
        )
        assert result == [1, 0]

    def test_fills_missing_indices(self):
        result = StylistJudge._parse_outfit_ranking(
            '{"ranking": [3]}', n_outfits=4,
        )
        assert result[0] == 2  # winner
        assert set(result) == {0, 1, 2, 3}  # all present

    def test_empty_ranking_list(self):
        result = StylistJudge._parse_outfit_ranking('{"ranking": []}', n_outfits=3)
        assert result is None


class TestRankOutfits:

    def test_single_outfit_returns_none(self, judge):
        source = _p()
        outfits = [{"bottoms": _cand("b1")}]
        intent = FitIntent("casual", "balanced", "casual", "blouse", "mild")
        result = judge.rank_outfits(source, outfits, intent)
        assert result is None

    def test_empty_outfits_returns_none(self, judge):
        source = _p()
        intent = FitIntent("casual", "balanced", "casual", "blouse", "mild")
        result = judge.rank_outfits(source, [], intent)
        assert result is None

    def test_successful_ranking(self, judge):
        source = _p()
        outfits = [
            {"bottoms": _cand("b1"), "outerwear": _cand("o1")},
            {"bottoms": _cand("b2"), "outerwear": _cand("o2")},
            {"bottoms": _cand("b3"), "outerwear": _cand("o3")},
        ]
        intent = FitIntent("casual", "balanced", "casual", "blouse", "mild")
        judge.client.chat.completions.create.return_value = _mock_ranking_response([2, 3, 1])
        result = judge.rank_outfits(source, outfits, intent)
        assert result == [1, 2, 0]

    def test_api_failure_returns_none(self, judge):
        source = _p()
        outfits = [
            {"bottoms": _cand("b1")},
            {"bottoms": _cand("b2")},
        ]
        intent = FitIntent("casual", "balanced", "casual", "blouse", "mild")
        judge.client.chat.completions.create.side_effect = Exception("API down")
        result = judge.rank_outfits(source, outfits, intent)
        assert result is None

    def test_malformed_response_returns_none(self, judge):
        source = _p()
        outfits = [
            {"bottoms": _cand("b1")},
            {"bottoms": _cand("b2")},
        ]
        intent = FitIntent("casual", "balanced", "casual", "blouse", "mild")
        msg = MagicMock()
        msg.content = '{"oops": true}'
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        judge.client.chat.completions.create.return_value = resp
        result = judge.rank_outfits(source, outfits, intent)
        assert result is None

    def test_uses_json_mode(self, judge):
        source = _p()
        outfits = [
            {"bottoms": _cand("b1")},
            {"bottoms": _cand("b2")},
        ]
        intent = FitIntent("casual", "balanced", "casual", "blouse", "mild")
        judge.client.chat.completions.create.return_value = _mock_ranking_response([1, 2])
        judge.rank_outfits(source, outfits, intent)
        call_kwargs = judge.client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_invalid_source_url_returns_none(self, judge):
        source = _p(image_url="https://www.example.com/product.html")
        outfits = [
            {"tops": _cand("t1")},
            {"tops": _cand("t2")},
        ]
        intent = FitIntent("casual", "balanced", "casual", "pants", "mild")
        result = judge.rank_outfits(source, outfits, intent)
        assert result is None

    def test_filters_outfits_with_bad_urls(self, judge):
        """Outfits with invalid image URLs are excluded from ranking."""
        source = _p()
        good_outfit = {"bottoms": _cand("b1", image_url="https://cdn.example.com/b1.jpg")}
        bad_outfit = {"bottoms": _cand("b2", image_url="https://www.example.com/page.html")}
        outfits = [good_outfit, bad_outfit]
        intent = FitIntent("casual", "balanced", "casual", "blouse", "mild")
        # Only 1 valid outfit → returns None (need ≥2)
        result = judge.rank_outfits(source, outfits, intent)
        assert result is None


# ===========================================================================
# 12. ENGINE HELPERS: combo generation + ranking application
# ===========================================================================

from services.outfit_engine import (
    _generate_outfit_combos,
    _apply_outfit_ranking,
    _apply_category_ranking,
)


def _scored_entry(pid, tattoo=0.8, cat="bottoms"):
    """Build a minimal scored dict matching _score_category output."""
    profile = _cand(pid=pid, broad_category=cat)
    return {
        "profile": profile,
        "tattoo": tattoo,
        "compat": tattoo * 0.6,
        "cosine": tattoo * 0.4,
        "dims": {},
    }


class TestApplyCategoryRanking:

    def test_reorders_by_ranking(self):
        scored = [_scored_entry("c1", 0.9), _scored_entry("c2", 0.8), _scored_entry("c3", 0.7)]
        ranking = ["c3", "c1", "c2"]
        result = _apply_category_ranking(scored, ranking)
        assert [e["profile"].product_id for e in result] == ["c3", "c1", "c2"]

    def test_unranked_items_appended(self):
        scored = [_scored_entry("c1"), _scored_entry("c2"), _scored_entry("c3"), _scored_entry("c4")]
        ranking = ["c3", "c1"]
        result = _apply_category_ranking(scored, ranking)
        pids = [e["profile"].product_id for e in result]
        assert pids == ["c3", "c1", "c2", "c4"]

    def test_does_not_mutate_input(self):
        scored = [_scored_entry("c1"), _scored_entry("c2")]
        original_order = [e["profile"].product_id for e in scored]
        _apply_category_ranking(scored, ["c2", "c1"])
        assert [e["profile"].product_id for e in scored] == original_order

    def test_unknown_ids_in_ranking_ignored(self):
        scored = [_scored_entry("c1"), _scored_entry("c2")]
        ranking = ["c99", "c2", "c1"]
        result = _apply_category_ranking(scored, ranking)
        assert [e["profile"].product_id for e in result] == ["c2", "c1"]

    def test_empty_ranking_preserves_order(self):
        scored = [_scored_entry("c1"), _scored_entry("c2")]
        result = _apply_category_ranking(scored, [])
        assert [e["profile"].product_id for e in result] == ["c1", "c2"]

    def test_all_ranked(self):
        scored = [_scored_entry("c1"), _scored_entry("c2"), _scored_entry("c3")]
        ranking = ["c2", "c3", "c1"]
        result = _apply_category_ranking(scored, ranking)
        assert len(result) == 3
        assert [e["profile"].product_id for e in result] == ["c2", "c3", "c1"]


class TestGenerateOutfitCombos:

    def test_empty_input(self):
        assert _generate_outfit_combos({}) == []

    def test_single_category(self):
        scored = {"bottoms": [_scored_entry("b1"), _scored_entry("b2")]}
        combos = _generate_outfit_combos(scored, top_per_cat=2, max_combos=6)
        assert len(combos) == 2
        assert "bottoms" in combos[0]

    def test_two_categories_cartesian(self):
        scored = {
            "bottoms": [_scored_entry("b1", 0.9), _scored_entry("b2", 0.7)],
            "outerwear": [_scored_entry("o1", 0.85, "outerwear"), _scored_entry("o2", 0.6, "outerwear")],
        }
        combos = _generate_outfit_combos(scored, top_per_cat=2, max_combos=6)
        assert len(combos) == 4  # 2 x 2
        # Best combo should be the one with highest avg tattoo
        best = combos[0]
        assert best["bottoms"]["profile"].product_id == "b1"
        assert best["outerwear"]["profile"].product_id == "o1"

    def test_max_combos_capped(self):
        scored = {
            "bottoms": [_scored_entry(f"b{i}", 0.9 - i*0.05) for i in range(4)],
            "outerwear": [_scored_entry(f"o{i}", 0.8 - i*0.05, "outerwear") for i in range(4)],
        }
        combos = _generate_outfit_combos(scored, top_per_cat=4, max_combos=6)
        assert len(combos) == 6  # Capped at 6 from 16

    def test_top_per_cat_limits(self):
        scored = {
            "bottoms": [_scored_entry(f"b{i}") for i in range(10)],
            "outerwear": [_scored_entry(f"o{i}", cat="outerwear") for i in range(10)],
        }
        combos = _generate_outfit_combos(scored, top_per_cat=2, max_combos=6)
        assert len(combos) == 4  # 2 x 2

    def test_sorted_by_avg_tattoo_desc(self):
        scored = {
            "bottoms": [_scored_entry("b1", 0.5), _scored_entry("b2", 0.9)],
            "outerwear": [_scored_entry("o1", 0.5, "outerwear"), _scored_entry("o2", 0.9, "outerwear")],
        }
        combos = _generate_outfit_combos(scored, top_per_cat=2, max_combos=4)
        # Best avg = (0.9+0.9)/2=0.9, worst = (0.5+0.5)/2=0.5
        assert combos[0]["bottoms"]["profile"].product_id == "b2"
        assert combos[0]["outerwear"]["profile"].product_id == "o2"

    def test_three_categories(self):
        scored = {
            "bottoms": [_scored_entry("b1", 0.8)],
            "outerwear": [_scored_entry("o1", 0.7, "outerwear")],
            "footwear": [_scored_entry("f1", 0.6, "footwear")],
        }
        combos = _generate_outfit_combos(scored, top_per_cat=1, max_combos=6)
        assert len(combos) == 1
        assert set(combos[0].keys()) == {"bottoms", "outerwear", "footwear"}

    def test_empty_category_excluded(self):
        scored = {
            "bottoms": [_scored_entry("b1", 0.8)],
            "outerwear": [],
        }
        combos = _generate_outfit_combos(scored, top_per_cat=2, max_combos=6)
        assert len(combos) == 1
        assert "outerwear" not in combos[0]


class TestApplyOutfitRanking:

    def test_winner_promoted_to_top(self):
        b1 = _scored_entry("b1", 0.7)
        b2 = _scored_entry("b2", 0.9)
        o1 = _scored_entry("o1", 0.7, "outerwear")
        o2 = _scored_entry("o2", 0.9, "outerwear")

        scored_by_cat = {
            "bottoms": [b2, b1],   # b2 is TATTOO #1
            "outerwear": [o2, o1], # o2 is TATTOO #1
        }

        combos = [
            {"bottoms": b2, "outerwear": o2},  # combo 0: TATTOO favorite
            {"bottoms": b1, "outerwear": o1},  # combo 1
        ]

        # Vision says combo 1 (the underdog) is best
        ranking = [1, 0]
        _apply_outfit_ranking(scored_by_cat, combos, ranking)

        # b1 and o1 should now be at position 0
        assert scored_by_cat["bottoms"][0]["profile"].product_id == "b1"
        assert scored_by_cat["outerwear"][0]["profile"].product_id == "o1"

    def test_winner_already_on_top(self):
        b1 = _scored_entry("b1", 0.9)
        b2 = _scored_entry("b2", 0.7)

        scored_by_cat = {"bottoms": [b1, b2]}
        combos = [{"bottoms": b1}, {"bottoms": b2}]
        ranking = [0, 1]  # TATTOO order confirmed
        _apply_outfit_ranking(scored_by_cat, combos, ranking)
        assert scored_by_cat["bottoms"][0]["profile"].product_id == "b1"

    def test_empty_ranking_no_change(self):
        b1 = _scored_entry("b1", 0.9)
        scored_by_cat = {"bottoms": [b1]}
        combos = [{"bottoms": b1}]
        _apply_outfit_ranking(scored_by_cat, combos, [])
        assert scored_by_cat["bottoms"][0]["profile"].product_id == "b1"

    def test_preserves_rest_of_list(self):
        entries = [_scored_entry(f"b{i}", 0.9 - i*0.1) for i in range(5)]
        scored_by_cat = {"bottoms": list(entries)}
        combos = [{"bottoms": entries[0]}, {"bottoms": entries[3]}]
        # Vision picks combo 1 (entry[3]) as best
        ranking = [1, 0]
        _apply_outfit_ranking(scored_by_cat, combos, ranking)
        # entries[3] promoted to top, rest still present
        assert scored_by_cat["bottoms"][0]["profile"].product_id == "b3"
        assert len(scored_by_cat["bottoms"]) == 5


# ===========================================================================
# 13. Image URL Validation
# ===========================================================================

class TestImageUrlValidation:
    """Tests for _is_valid_image_url()."""

    # --- Valid URLs ---

    def test_jpg_extension(self):
        url = "https://usepose.s3.us-east-1.amazonaws.com/products/test/primary.jpg"
        assert _is_valid_image_url(url) is True

    def test_jpeg_extension(self):
        url = "https://cdn.example.com/image.jpeg"
        assert _is_valid_image_url(url) is True

    def test_png_extension(self):
        url = "https://cdn.example.com/image.png"
        assert _is_valid_image_url(url) is True

    def test_webp_extension(self):
        url = "https://cdn.example.com/image.webp"
        assert _is_valid_image_url(url) is True

    def test_gif_extension(self):
        url = "https://cdn.example.com/image.gif"
        assert _is_valid_image_url(url) is True

    def test_s3_url_with_path(self):
        """S3 URLs are our product images — always valid."""
        url = "https://usepose.s3.us-east-1.amazonaws.com/products/abc/original_0_"
        assert _is_valid_image_url(url) is True

    def test_cloudfront_url(self):
        """CloudFront CDN URLs are valid."""
        url = "https://d1234.cloudfront.net/images/product.jpg"
        assert _is_valid_image_url(url) is True

    def test_scene7_with_fmt_jpeg(self):
        """Dynamic image CDN with fmt=jpeg param."""
        url = "https://s7d2.scene7.com/is/image/aeo/0577_9097_100_f?$plp-web-desktop$&fmt=jpeg"
        assert _is_valid_image_url(url) is True

    def test_scene7_with_fmt_png(self):
        url = "https://s7d2.scene7.com/is/image/aeo/test?fmt=png"
        assert _is_valid_image_url(url) is True

    def test_jpg_with_query_string(self):
        """JPG with query params should still be valid."""
        url = "https://cdn.example.com/image.jpg?w=400&h=600"
        assert _is_valid_image_url(url) is True

    # --- Invalid URLs ---

    def test_html_page_url(self):
        """Aritzia product page URLs (.html) are NOT images."""
        url = "https://www.aritzia.com/us/en/product/encourage-poplin-dress/131094-14396.html#main"
        assert _is_valid_image_url(url) is False

    def test_html_without_fragment(self):
        url = "https://www.example.com/product/details.html"
        assert _is_valid_image_url(url) is False

    def test_htm_extension(self):
        url = "https://www.example.com/page.htm"
        assert _is_valid_image_url(url) is False

    def test_php_page(self):
        url = "https://www.example.com/image.php?id=123"
        assert _is_valid_image_url(url) is False

    def test_empty_string(self):
        assert _is_valid_image_url("") is False

    def test_none_like(self):
        assert _is_valid_image_url("") is False

    def test_non_http(self):
        assert _is_valid_image_url("ftp://example.com/img.jpg") is False

    def test_no_extension_no_hints(self):
        """Dynamic URL with no extension and no format hint - invalid."""
        url = "https://www.aritzia.com/is/image/Aritzia/13331036342_09?wid=1200&hei=1800"
        assert _is_valid_image_url(url) is False

    # --- Edge cases ---

    def test_uppercase_extension(self):
        """Case-insensitive extension check."""
        url = "https://cdn.example.com/image.JPG"
        assert _is_valid_image_url(url) is True

    def test_s3_url_no_extension(self):
        """S3 URL without extension is still valid (it's our CDN)."""
        url = "https://usepose.s3.us-east-1.amazonaws.com/products/abc/original_0_"
        assert _is_valid_image_url(url) is True


class TestJudgeUrlFiltering:
    """Tests that the stylist judge pre-filters invalid image URLs."""

    def test_build_rerank_messages_skips_bad_source(self):
        """Source with .html URL - no image_url in messages."""
        source = _p(
            image_url="https://www.aritzia.com/product/test.html#main",
        )
        cand = _cand("c1", image_url="https://cdn.example.com/c1.jpg")
        intent = FitIntent("casual", "balanced", "casual", "blouse", "mild")
        msgs = build_rerank_messages(source, [cand], intent)
        user_content = msgs[1]["content"]
        image_parts = [p for p in user_content if p.get("type") == "image_url"]
        # Only the candidate image should be present
        assert len(image_parts) == 1
        assert "c1.jpg" in image_parts[0]["image_url"]["url"]

    def test_build_rerank_messages_skips_bad_candidate(self):
        """Candidate with .html URL - no image_url for that candidate."""
        source = _p(image_url="https://cdn.example.com/src.jpg")
        good_cand = _cand("c1", image_url="https://cdn.example.com/c1.jpg")
        bad_cand = _cand("c2", image_url="https://www.aritzia.com/product/bad.html")
        intent = FitIntent("casual", "balanced", "casual", "blouse", "mild")
        msgs = build_rerank_messages(source, [good_cand, bad_cand], intent)
        user_content = msgs[1]["content"]
        image_parts = [p for p in user_content if p.get("type") == "image_url"]
        # Source + good candidate = 2 images
        assert len(image_parts) == 2

    def test_rerank_category_skips_invalid_source(self):
        """rerank_category returns None when source URL is invalid."""
        with patch("openai.OpenAI"):
            judge = StylistJudge(api_key="test-key")
        source = _p(image_url="https://www.aritzia.com/product/test.html#main")
        cand = _cand("c1", image_url="https://cdn.example.com/c1.jpg")
        intent = FitIntent("casual", "balanced", "casual", "blouse", "mild")
        result = judge.rerank_category(source, [cand], intent)
        assert result is None

    def test_rerank_category_filters_bad_candidates(self):
        """rerank_category filters out candidates with bad image URLs."""
        with patch("openai.OpenAI"):
            judge = StylistJudge(api_key="test-key")
        source = _p(image_url="https://cdn.example.com/src.jpg")
        good_cand = _cand("c1", image_url="https://cdn.example.com/c1.jpg")
        bad_cand = _cand("c2", image_url="https://www.example.com/page.html")
        intent = FitIntent("casual", "balanced", "casual", "blouse", "mild")
        with patch.object(judge, "_call_rerank", return_value=["c1"]):
            result = judge.rerank_category(source, [good_cand, bad_cand], intent)
        assert result is not None
        assert "c1" in result

    def test_rank_outfits_skips_invalid_source(self):
        """rank_outfits returns None when source URL is invalid."""
        with patch("openai.OpenAI"):
            judge = StylistJudge(api_key="test-key")
        source = _p(image_url="https://www.example.com/product.html")
        outfits = [
            {"tops": _cand("t1")},
            {"tops": _cand("t2")},
        ]
        intent = FitIntent("casual", "balanced", "casual", "pants", "mild")
        result = judge.rank_outfits(source, outfits, intent)
        assert result is None


# ===========================================================================
# 14. AVIF Detection
# ===========================================================================

class TestAvifDetection:
    """Tests for _is_supported_image_format() AVIF magic-byte detection."""

    def test_avif_constants_defined(self):
        assert _AVIF_FTYP == b"ftyp"
        assert b"avif" in _AVIF_BRANDS
        assert b"mif1" in _AVIF_BRANDS

    def test_avif_detected_via_mock(self):
        """AVIF file bytes → detected as unsupported."""
        # Build minimal AVIF header: [size(4)][ftyp(4)][brand(4)]
        avif_header = b"\x00\x00\x00\x1c" + b"ftyp" + b"avif" + b"\x00" * 20
        with patch("services.outfit_judge._format_cache", {}):
            with patch("requests.get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.status_code = 206
                mock_resp.content = avif_header
                mock_get.return_value = mock_resp
                assert _is_supported_image_format("https://cdn.example.com/img.jpg") is False

    def test_jpeg_passes(self):
        """Real JPEG header → supported."""
        jpeg_header = b"\xff\xd8\xff\xe0" + b"\x00" * 28
        with patch("services.outfit_judge._format_cache", {}):
            with patch("requests.get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.status_code = 206
                mock_resp.content = jpeg_header
                mock_get.return_value = mock_resp
                assert _is_supported_image_format("https://cdn.example.com/img.jpg") is True

    def test_png_passes(self):
        """PNG header → supported."""
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
        with patch("services.outfit_judge._format_cache", {}):
            with patch("requests.get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.content = png_header
                mock_get.return_value = mock_resp
                assert _is_supported_image_format("https://cdn.example.com/img.png") is True

    def test_network_error_optimistic(self):
        """Network error → optimistic True (don't block all candidates)."""
        with patch("services.outfit_judge._format_cache", {}):
            with patch("requests.get", side_effect=Exception("timeout")):
                assert _is_supported_image_format("https://cdn.example.com/img.jpg") is True

    def test_empty_url_returns_false(self):
        assert _is_supported_image_format("") is False

    def test_cache_hit(self):
        """Cached result is returned without network call."""
        url = "https://cdn.example.com/cached.jpg"
        with patch("services.outfit_judge._format_cache", {url: False}):
            # No requests.get mock — would fail if called
            assert _is_supported_image_format(url) is False

    def test_mif1_brand_detected(self):
        """AVIF with mif1 brand → unsupported."""
        avif_header = b"\x00\x00\x00\x1c" + b"ftyp" + b"mif1" + b"\x00" * 20
        with patch("services.outfit_judge._format_cache", {}):
            with patch("requests.get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.status_code = 206
                mock_resp.content = avif_header
                mock_get.return_value = mock_resp
                assert _is_supported_image_format("https://cdn.example.com/img.jpg") is False

    def test_short_response_optimistic(self):
        """Response too short to check → optimistic True."""
        with patch("services.outfit_judge._format_cache", {}):
            with patch("requests.get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.status_code = 206
                mock_resp.content = b"\xff\xd8"  # only 2 bytes
                mock_get.return_value = mock_resp
                assert _is_supported_image_format("https://cdn.example.com/img.jpg") is True
