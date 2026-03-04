"""Unit tests for the Vision Pass/Fail Judge (Complete the Fit v3).

Uses SimpleNamespace mocks — no real OpenAI API key needed.
All vision API calls are mocked.
"""

import json
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from services.outfit_judge import (
    FitIntent,
    VisionJudge,
    _SYSTEM_PROMPT,
    _OUTFIT_RANKING_PROMPT,
    _OCCASION_TO_BUCKET,
    _FITTED_SILS,
    _OVERSIZED_SILS,
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


def _mock_vision_response(fail_ids):
    """Build a mock OpenAI vision response with a fail list."""
    content = json.dumps({"fail": fail_ids})
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
        j = VisionJudge(api_key="test-key", model="gpt-4o-mini", timeout=5.0)
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
# 3. VISION MESSAGE BUILDING
# ===========================================================================

class TestVisionMessages:

    def test_message_structure(self):
        msgs = build_vision_messages(_p(), [_cand("c1")], FitIntent("casual", "balanced", "casual", "T-Shirt", "mild"))
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert isinstance(msgs[1]["content"], list)

    def test_source_image_included(self):
        src = _p(image_url="https://cdn.example.com/source.jpg")
        msgs = build_vision_messages(src, [_cand("c1")], FitIntent("casual", "balanced", "casual", "T-Shirt", "mild"))
        image_parts = [p for p in msgs[1]["content"] if p.get("type") == "image_url"]
        urls = [p["image_url"]["url"] for p in image_parts]
        assert "https://cdn.example.com/source.jpg" in urls

    def test_candidate_images_included(self):
        c1 = _cand("c1")
        c2 = _cand("c2")
        msgs = build_vision_messages(_p(), [c1, c2], FitIntent("casual", "balanced", "casual", "T-Shirt", "mild"))
        image_parts = [p for p in msgs[1]["content"] if p.get("type") == "image_url"]
        # 1 source + 2 candidates = 3 images
        assert len(image_parts) == 3

    def test_all_images_low_detail(self):
        msgs = build_vision_messages(_p(), [_cand("c1")], FitIntent("casual", "balanced", "casual", "T-Shirt", "mild"))
        image_parts = [p for p in msgs[1]["content"] if p.get("type") == "image_url"]
        for part in image_parts:
            assert part["image_url"]["detail"] == "low"

    def test_candidate_id_in_text(self):
        msgs = build_vision_messages(_p(), [_cand("c1")], FitIntent("casual", "balanced", "casual", "T-Shirt", "mild"))
        text_parts = [p["text"] for p in msgs[1]["content"] if p.get("type") == "text"]
        full_text = " ".join(text_parts)
        assert "c1" in full_text

    def test_context_line_in_message(self):
        intent = FitIntent("work", "fitted", "classic", "Blouse", "mild")
        msgs = build_vision_messages(_p(), [_cand("c1")], intent)
        text_parts = [p["text"] for p in msgs[1]["content"] if p.get("type") == "text"]
        full_text = " ".join(text_parts)
        assert "work" in full_text

    def test_no_name_or_brand_in_message(self):
        src = _p(name="My Fancy Blouse", brand="Gucci")
        msgs = build_vision_messages(src, [_cand("c1")], FitIntent("casual", "balanced", "casual", "T-Shirt", "mild"))
        text_parts = [p["text"] for p in msgs[1]["content"] if p.get("type") == "text"]
        full_text = " ".join(text_parts)
        assert "My Fancy Blouse" not in full_text
        assert "Gucci" not in full_text

    def test_no_price_in_message(self):
        src = _p(price=199.99)
        msgs = build_vision_messages(src, [_cand("c1")], FitIntent("casual", "balanced", "casual", "T-Shirt", "mild"))
        text_parts = [p["text"] for p in msgs[1]["content"] if p.get("type") == "text"]
        full_text = " ".join(text_parts)
        assert "199" not in full_text

    def test_missing_source_image_skips_image_part(self):
        src = _p(image_url="")
        msgs = build_vision_messages(src, [_cand("c1")], FitIntent("casual", "balanced", "casual", "T-Shirt", "mild"))
        image_parts = [p for p in msgs[1]["content"] if p.get("type") == "image_url"]
        # Only candidate image, no source image
        assert len(image_parts) == 1

    def test_missing_candidate_image_skips_image_part(self):
        c = _cand("c1")
        c.image_url = ""
        msgs = build_vision_messages(_p(), [c], FitIntent("casual", "balanced", "casual", "T-Shirt", "mild"))
        image_parts = [p for p in msgs[1]["content"] if p.get("type") == "image_url"]
        # Only source image, no candidate image
        assert len(image_parts) == 1


# ===========================================================================
# 4. SYSTEM PROMPT
# ===========================================================================

class TestSystemPrompt:

    def test_clash_definition_present(self):
        assert "CLASH" in _SYSTEM_PROMPT

    def test_pass_bias_present(self):
        assert "PASS" in _SYSTEM_PROMPT
        assert "When in doubt" in _SYSTEM_PROMPT

    def test_json_schema_present(self):
        assert '"fail"' in _SYSTEM_PROMPT

    def test_no_scoring_rubric(self):
        """No weighted scoring dimensions — this is pass/fail only."""
        assert "weight 0.25" not in _SYSTEM_PROMPT
        assert "SCORING RUBRIC" not in _SYSTEM_PROMPT


# ===========================================================================
# 5. RESPONSE PARSING
# ===========================================================================

class TestResponseParsing:

    def test_valid_fail_list(self):
        content = json.dumps({"fail": ["c1", "c3"]})
        assert VisionJudge._parse_response(content) == {"c1", "c3"}

    def test_empty_fail_list(self):
        content = json.dumps({"fail": []})
        assert VisionJudge._parse_response(content) == set()

    def test_malformed_json(self):
        assert VisionJudge._parse_response("not json {{{") == set()

    def test_none_content(self):
        assert VisionJudge._parse_response(None) == set()

    def test_missing_fail_key(self):
        content = json.dumps({"something": "else"})
        assert VisionJudge._parse_response(content) == set()

    def test_fail_not_list(self):
        content = json.dumps({"fail": "c1"})
        assert VisionJudge._parse_response(content) == set()

    def test_fail_with_nulls_filtered(self):
        content = json.dumps({"fail": ["c1", None, "", "c2"]})
        parsed = VisionJudge._parse_response(content)
        assert parsed == {"c1", "c2"}


# ===========================================================================
# 6. JUDGE BATCH
# ===========================================================================

class TestJudgeBatch:

    def test_empty_candidates(self, judge):
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        assert judge.judge_batch(_p(), [], intent) == set()

    def test_all_pass(self, judge):
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        judge.client.chat.completions.create.return_value = _mock_vision_response([])
        result = judge.judge_batch(_p(), [_cand("c1"), _cand("c2")], intent)
        assert result == set()

    def test_some_fail(self, judge):
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        judge.client.chat.completions.create.return_value = _mock_vision_response(["c2"])
        result = judge.judge_batch(_p(), [_cand("c1"), _cand("c2"), _cand("c3")], intent)
        assert result == {"c2"}

    def test_all_fail(self, judge):
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        judge.client.chat.completions.create.return_value = _mock_vision_response(["c1", "c2"])
        result = judge.judge_batch(_p(), [_cand("c1"), _cand("c2")], intent)
        assert result == {"c1", "c2"}

    def test_unknown_id_in_fail_ignored(self, judge):
        """If LLM returns an ID not in the candidates, it's stored in cache but doesn't cause errors."""
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        judge.client.chat.completions.create.return_value = _mock_vision_response(["unknown_id"])
        result = judge.judge_batch(_p(), [_cand("c1")], intent)
        # c1 is not in the fail list, so passes
        assert "c1" not in result


# ===========================================================================
# 7. CACHE
# ===========================================================================

class TestCache:

    def test_cache_hit_skips_llm(self, judge):
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        judge.client.chat.completions.create.return_value = _mock_vision_response(["c1"])
        judge.judge_batch(_p(), [_cand("c1")], intent)
        assert judge.client.chat.completions.create.call_count == 1
        # Second call — cached
        judge.judge_batch(_p(), [_cand("c1")], intent)
        assert judge.client.chat.completions.create.call_count == 1

    def test_cached_fail_remembered(self, judge):
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        judge.client.chat.completions.create.return_value = _mock_vision_response(["c1"])
        judge.judge_batch(_p(), [_cand("c1")], intent)
        # Second call — c1 still fails from cache
        result = judge.judge_batch(_p(), [_cand("c1")], intent)
        assert result == {"c1"}

    def test_cached_pass_remembered(self, judge):
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        judge.client.chat.completions.create.return_value = _mock_vision_response([])
        judge.judge_batch(_p(), [_cand("c1")], intent)
        result = judge.judge_batch(_p(), [_cand("c1")], intent)
        assert result == set()

    def test_cache_miss_different_source(self, judge):
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        judge.client.chat.completions.create.return_value = _mock_vision_response([])
        judge.judge_batch(_p(product_id="src-a"), [_cand("c1")], intent)
        judge.judge_batch(_p(product_id="src-b"), [_cand("c1")], intent)
        assert judge.client.chat.completions.create.call_count == 2

    def test_mixed_cached_and_uncached(self, judge):
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        # Cache c1
        judge.client.chat.completions.create.return_value = _mock_vision_response(["c1"])
        judge.judge_batch(_p(), [_cand("c1")], intent)
        # Batch with c1 (cached fail) + c2 (uncached)
        judge.client.chat.completions.create.return_value = _mock_vision_response([])
        result = judge.judge_batch(_p(), [_cand("c1"), _cand("c2")], intent)
        assert judge.client.chat.completions.create.call_count == 2
        assert result == {"c1"}  # c1 from cache, c2 passes

    def test_cache_eviction(self, judge):
        judge._CACHE_SIZE = 2
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        judge.client.chat.completions.create.return_value = _mock_vision_response([])
        judge.judge_batch(_p(), [_cand("c0")], intent)
        judge.judge_batch(_p(), [_cand("c1")], intent)
        assert len(judge._cache) == 2
        # Add c2 -> evicts c0
        judge.judge_batch(_p(), [_cand("c2")], intent)
        assert len(judge._cache) == 2
        # c0 evicted -> needs new call
        calls_before = judge.client.chat.completions.create.call_count
        judge.judge_batch(_p(), [_cand("c0")], intent)
        assert judge.client.chat.completions.create.call_count > calls_before


# ===========================================================================
# 8. GRACEFUL DEGRADATION
# ===========================================================================

class TestGracefulDegradation:

    def test_exception_returns_empty(self, judge):
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        judge.client.chat.completions.create.side_effect = Exception("timeout")
        result = judge.judge_batch(_p(), [_cand("c1"), _cand("c2")], intent)
        assert result == set()  # keep all

    def test_retry_on_failure(self, judge):
        judge.max_retries = 1
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        good = _mock_vision_response(["c1"])
        judge.client.chat.completions.create.side_effect = [Exception("err"), good]
        with patch("services.outfit_judge._time.sleep"):
            result = judge.judge_batch(_p(), [_cand("c1")], intent)
        assert result == {"c1"}
        assert judge.client.chat.completions.create.call_count == 2

    def test_all_retries_exhausted(self, judge):
        judge.max_retries = 1
        intent = FitIntent("casual", "balanced", "casual", "T-Shirt", "mild")
        judge.client.chat.completions.create.side_effect = Exception("persistent")
        with patch("services.outfit_judge._time.sleep"):
            result = judge.judge_batch(_p(), [_cand("c1")], intent)
        assert result == set()  # keep all

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
            mock_settings.llm_judge_model = "gpt-4o-mini"
            mock_settings.llm_judge_timeout = 20.0
            with patch("config.settings.get_settings", return_value=mock_settings), \
                 patch("openai.OpenAI"):
                result = get_pair_judge()
                assert isinstance(result, VisionJudge)
        finally:
            oj._judge = original


# ===========================================================================
# 9. OCCASION BUCKETS + SILHOUETTE SETS
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
# OUTFIT RANKING
# ===========================================================================

def _mock_ranking_response(ranking):
    """Build a mock OpenAI response with a ranking list."""
    content = json.dumps({"ranking": ranking})
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _outfit(cats_and_pids):
    """Build a mock outfit dict: {cat: AestheticProfile, ...}."""
    return {
        cat: _cand(pid=pid, broad_category=cat)
        for cat, pid in cats_and_pids.items()
    }


class TestOutfitRankingPrompt:

    def test_prompt_rewards_contrast(self):
        assert "contrast" in _OUTFIT_RANKING_PROMPT.lower()

    def test_prompt_penalizes_boring(self):
        lower = _OUTFIT_RANKING_PROMPT.lower()
        assert "boring" in lower or "mediocre" in lower

    def test_prompt_requests_json(self):
        assert '{"ranking":' in _OUTFIT_RANKING_PROMPT

    def test_prompt_ranks_all(self):
        assert "Rank ALL" in _OUTFIT_RANKING_PROMPT


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

    def test_system_prompt_is_ranking_prompt(self):
        source = _p()
        outfits = self._make_outfits()
        intent = FitIntent("casual", "balanced", "casual", "blouse", "mild")
        msgs = build_outfit_ranking_messages(source, outfits, intent)
        assert msgs[0]["content"] == _OUTFIT_RANKING_PROMPT

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

    def test_all_outfit_images_low_detail(self):
        source = _p()
        outfits = self._make_outfits()
        intent = FitIntent("casual", "balanced", "casual", "blouse", "mild")
        msgs = build_outfit_ranking_messages(source, outfits, intent)
        images = [
            p for p in msgs[1]["content"]
            if isinstance(p, dict) and p.get("type") == "image_url"
        ]
        for img in images:
            assert img["image_url"]["detail"] == "low"

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
        """Source + 2 outfits × 2 pieces = 5 images."""
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

    def test_context_line_in_message(self):
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


class TestParseRankingResponse:

    def test_valid_ranking(self):
        result = VisionJudge._parse_ranking_response(
            '{"ranking": [3, 1, 2]}', n_outfits=3,
        )
        assert result == [2, 0, 1]  # 0-indexed

    def test_single_winner(self):
        result = VisionJudge._parse_ranking_response(
            '{"ranking": [2]}', n_outfits=3,
        )
        # [1] + fill in [0, 2]
        assert result[0] == 1  # winner is outfit index 1

    def test_invalid_json(self):
        result = VisionJudge._parse_ranking_response("not json", n_outfits=3)
        assert result is None

    def test_not_a_dict(self):
        result = VisionJudge._parse_ranking_response("[1, 2, 3]", n_outfits=3)
        assert result is None

    def test_no_ranking_key(self):
        result = VisionJudge._parse_ranking_response('{"result": [1, 2]}', n_outfits=2)
        assert result is None

    def test_ranking_not_a_list(self):
        result = VisionJudge._parse_ranking_response('{"ranking": "1,2,3"}', n_outfits=3)
        assert result is None

    def test_out_of_range_filtered(self):
        result = VisionJudge._parse_ranking_response(
            '{"ranking": [1, 99, 2]}', n_outfits=2,
        )
        assert result == [0, 1]

    def test_duplicates_filtered(self):
        result = VisionJudge._parse_ranking_response(
            '{"ranking": [2, 2, 1]}', n_outfits=2,
        )
        assert result == [1, 0]

    def test_fills_missing_indices(self):
        result = VisionJudge._parse_ranking_response(
            '{"ranking": [3]}', n_outfits=4,
        )
        assert result[0] == 2  # winner
        assert set(result) == {0, 1, 2, 3}  # all present

    def test_empty_ranking_list(self):
        result = VisionJudge._parse_ranking_response('{"ranking": []}', n_outfits=3)
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
        # Return garbage JSON
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


# ===========================================================================
# ENGINE HELPERS: combo generation + ranking application
# ===========================================================================

from services.outfit_engine import _generate_outfit_combos, _apply_outfit_ranking


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
        assert len(combos) == 4  # 2 × 2
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
        assert len(combos) == 4  # 2 × 2

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
