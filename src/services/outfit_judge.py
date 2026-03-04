"""
LLM Pair Judge for Complete the Fit v3
=======================================

Reranks outfit complement candidates using a structured LLM evaluation.
Operates on the top-K candidates (by TATTOO score) from each target
category, producing a 0-10 overall score per candidate that blends
with the existing TATTOO score for final ranking.

Components:
  - FitIntent: derived from source product, describes what the outfit needs
  - derive_fit_intent(): source AestheticProfile -> FitIntent
  - LLMPairJudge: singleton OpenAI client with bounded LRU cache
  - JudgeResult: structured output per candidate
  - get_pair_judge(): thread-safe singleton accessor

Integration:
  Called from outfit_engine._score_category() after TATTOO scoring
  and before MMR diversity selection.
"""

import hashlib
import json
import logging
import threading
import time as _time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from services.outfit_engine import AestheticProfile

logger = logging.getLogger(__name__)


# =============================================================================
# 1. FIT INTENT  (derived from source product)
# =============================================================================

_OCCASION_TO_BUCKET = {
    "workout": "active", "gym": "active", "exercise": "active",
    "training": "active", "yoga": "active", "running": "active",
    "hiking": "active", "sports": "active",
    "work": "work", "office": "work", "business": "work",
    "meeting": "work", "interview": "work", "professional": "work",
    "date night": "going-out", "date": "going-out", "party": "going-out",
    "club": "going-out", "clubbing": "going-out", "night out": "going-out",
    "bar": "going-out", "cocktail": "going-out", "drinks": "going-out",
    "dinner": "going-out", "concert": "going-out",
    "wedding": "event", "wedding guest": "event", "formal event": "event",
    "gala": "event", "graduation": "event", "prom": "event",
    "ceremony": "event", "black tie": "event",
    "everyday": "casual", "casual": "casual", "weekend": "casual",
    "brunch": "casual", "shopping": "casual", "errands": "casual",
    "travel": "casual", "vacation": "casual", "lounging": "casual",
    "beach": "casual",
}

_FITTED_SILS = {"fitted", "slim", "bodycon", "tailored"}
_OVERSIZED_SILS = {"oversized", "relaxed", "boxy"}


@dataclass(frozen=True)
class FitIntent:
    """Describes what an outfit complement should achieve."""
    occasion_target: str            # work | casual | going-out | event | active
    silhouette_intent: str          # balanced | oversized | fitted | layered
    vibe: str                       # primary style tag
    statement_target: float         # 0.0-1.0
    warmth_target: str              # hot | mild | cold | any
    source_is_basic: bool
    source_statement_level: float

    def cache_key(self) -> str:
        """Short hash for cache keying — buckets statement_target to 0.1."""
        bucket = round(self.statement_target, 1)
        raw = f"{self.occasion_target}:{self.silhouette_intent}:{self.vibe}:{bucket}:{self.warmth_target}"
        return hashlib.md5(raw.encode()).hexdigest()[:8]


def derive_fit_intent(source: "AestheticProfile") -> FitIntent:
    """Derive outfit intent from a source product's AestheticProfile."""
    # Occasion target
    occasion_counts: Dict[str, int] = {}
    for occ in (source.occasions or []):
        bucket = _OCCASION_TO_BUCKET.get(occ.lower().strip(), "casual")
        occasion_counts[bucket] = occasion_counts.get(bucket, 0) + 1
    occasion_target = max(occasion_counts, key=occasion_counts.get) if occasion_counts else "casual"

    # Silhouette intent
    sil = (source.silhouette or "").lower().strip()
    l1 = (source.gemini_category_l1 or "").lower().strip()
    if sil in _OVERSIZED_SILS:
        silhouette_intent = "oversized"
    elif sil in _FITTED_SILS:
        silhouette_intent = "fitted"
    elif l1 == "outerwear":
        silhouette_intent = "layered"
    else:
        silhouette_intent = "balanced"

    # Vibe
    vibe = (source.primary_style or "casual").lower().strip()

    # Statement target (inverse of source)
    src_strength = source.style_strength
    source_is_basic = src_strength < 0.35
    if source_is_basic:
        statement_target = 0.5 + (0.35 - src_strength) * 0.5
    elif src_strength >= 0.70:
        statement_target = max(0.10, 0.40 - (src_strength - 0.70) * 1.0)
    else:
        statement_target = 0.40

    # Warmth target
    warmth_target = source.temp_band or "mild"

    return FitIntent(
        occasion_target=occasion_target,
        silhouette_intent=silhouette_intent,
        vibe=vibe,
        statement_target=round(statement_target, 2),
        warmth_target=warmth_target,
        source_is_basic=source_is_basic,
        source_statement_level=round(src_strength, 2),
    )


# =============================================================================
# 2. JUDGE RESULT
# =============================================================================

@dataclass
class JudgeResult:
    """Structured output from the LLM pair judge for one candidate."""
    overall: float          # 0-10
    fail: bool              # True = hard disqualification
    tags: List[str] = field(default_factory=list)


# =============================================================================
# 3. PROMPT CONSTRUCTION
# =============================================================================

_SYSTEM_PROMPT = (
    "You are a strict fashion outfit compatibility evaluator. You score how well "
    "each candidate garment complements a source garment to form a complete outfit.\n\n"
    "Return ONLY valid JSON matching this exact schema:\n"
    '{"results": [{"id": "<candidate_id>", "overall": <0-10>, "fail": <true|false>, '
    '"tags": ["<tag>", ...]}]}\n\n'
    "SCORING RUBRIC (apply strictly, overall = weighted blend of these):\n"
    "- Occasion/formality alignment (weight 0.25): same occasion bucket = good, "
    "gym+formal = fail\n"
    "- Silhouette balance (weight 0.20): fitted+wide = good, oversized+oversized = bad, "
    "consider the silhouette_intent\n"
    "- Fabric coherence (weight 0.20): texture contrast is good (denim+silk, knit+leather), "
    "same-material same-color = bad\n"
    "- Style coherence (weight 0.15): adjacent styles = good (classic+minimalist), "
    "clash = bad (romantic+sporty)\n"
    "- Color harmony (weight 0.10): neutral+color = good, competing brights = bad, "
    "do NOT default neutrals to high scores\n"
    "- Pattern balance (weight 0.10): solid+print = good, print+print = bad, "
    "two solids = acceptable but not exciting\n\n"
    "HARD FAILS (set fail=true, overall=0):\n"
    "- Strong occasion mismatch (gym vs formal, loungewear vs office event)\n"
    "- Near-identical item to source (same category_l2 + same color + same fabric)\n\n"
    "ANTI-NEUTRALITY RULE:\n"
    "- If the source is basic (statement_level < 0.35), penalize candidates that are "
    "also plain/neutral/basic UNLESS they are a perfect occasion+silhouette match. "
    "Score all-basic combos no higher than 5.\n"
    "- A 'basic' item: solid pattern, neutral color, low statement_level, no texture interest.\n\n"
    "STATEMENT BALANCE RULE:\n"
    "- If the source is a statement piece (statement_level > 0.70), prefer basic/supporting "
    "complements. Score two competing statement pieces no higher than 4.\n"
    "- Respect the statement_target: higher = wants more visual interest, lower = wants basics.\n\n"
    "TAGS (include 1-3 from this list):\n"
    "occasion_match, silhouette_balance, fabric_contrast, style_coherence, "
    "color_harmony, too_basic, too_busy, occasion_clash, great_complement, "
    "statement_conflict, good_contrast, season_mismatch"
)


def _profile_to_judge_dict(profile: "AestheticProfile") -> Dict[str, Any]:
    """Extract only the attributes the LLM judge needs. No names, brands, or prices."""
    return {
        "category_l2": profile.gemini_category_l2 or "",
        "formality": profile.formality or "",
        "silhouette": profile.silhouette or "",
        "fabric": profile.material_family or profile.apparent_fabric or "",
        "color": profile.color_family or profile.primary_color or "",
        "pattern": profile.pattern or "",
        "statement_level": round(profile.style_strength, 2),
        "occasions": (profile.occasions or [])[:3],
        "seasons": (profile.seasons or [])[:3],
        "texture": profile.texture or "",
        "sheen": profile.sheen or "",
        "length": profile.length or "",
    }


def build_judge_payload(
    source: "AestheticProfile",
    candidates: List["AestheticProfile"],
    intent: FitIntent,
) -> Dict[str, Any]:
    """Build the structured user-message payload for the LLM judge."""
    return {
        "intent": {
            "occasion": intent.occasion_target,
            "silhouette": intent.silhouette_intent,
            "vibe": intent.vibe,
            "statement_target": intent.statement_target,
            "warmth": intent.warmth_target,
        },
        "source": _profile_to_judge_dict(source),
        "candidates": [
            {"id": c.product_id or f"cand_{i}", **_profile_to_judge_dict(c)}
            for i, c in enumerate(candidates)
        ],
    }


# =============================================================================
# 4. LLM PAIR JUDGE
# =============================================================================

class LLMPairJudge:
    """Reranks outfit candidates using a structured LLM evaluation.

    Singleton — accessed via get_pair_judge().
    Uses the same OpenAI API key as the query planner.
    Bounded in-memory LRU cache for results.
    """

    _CACHE_SIZE = 256

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1-mini",
        timeout: float = 15.0,
        max_retries: int = 1,
    ):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        self._cache: Dict[str, JudgeResult] = {}
        self._cache_order: List[str] = []
        self._cache_lock = threading.Lock()

        logger.info(
            "LLMPairJudge initialized: model=%s, timeout=%.1fs, cache_size=%d",
            model, timeout, self._CACHE_SIZE,
        )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_cache_key(source_pid: str, cand_pid: str, intent_hash: str) -> str:
        return f"{source_pid}:{cand_pid}:{intent_hash}"

    def _cache_get(self, key: str) -> Optional[JudgeResult]:
        with self._cache_lock:
            return self._cache.get(key)

    def _cache_put(self, key: str, value: JudgeResult) -> None:
        with self._cache_lock:
            if key in self._cache:
                return
            if len(self._cache) >= self._CACHE_SIZE:
                evict = self._cache_order.pop(0)
                self._cache.pop(evict, None)
            self._cache[key] = value
            self._cache_order.append(key)

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def judge_batch(
        self,
        source: "AestheticProfile",
        candidates: List["AestheticProfile"],
        intent: FitIntent,
    ) -> Dict[str, JudgeResult]:
        """Score a batch of candidates against source with intent context.

        Returns {product_id: JudgeResult} for all candidates.
        On LLM failure, returns only cached results (graceful degradation).
        """
        if not candidates:
            return {}

        source_pid = source.product_id or ""
        intent_hash = intent.cache_key()

        # Separate cached from uncached
        results: Dict[str, JudgeResult] = {}
        uncached: List["AestheticProfile"] = []
        uncached_keys: List[str] = []

        for cand in candidates:
            cand_pid = cand.product_id or ""
            key = self._make_cache_key(source_pid, cand_pid, intent_hash)
            cached = self._cache_get(key)
            if cached is not None:
                results[cand_pid] = cached
            else:
                uncached.append(cand)
                uncached_keys.append(key)

        if not uncached:
            logger.debug("LLM judge: all %d candidates cached", len(candidates))
            return results

        logger.info(
            "LLM judge: %d cached, %d to evaluate for source %s",
            len(results), len(uncached), source_pid,
        )

        # Build payload and call LLM
        payload = build_judge_payload(source, uncached, intent)
        user_message = json.dumps(payload, separators=(",", ":"))

        llm_results = self._call_llm(user_message)

        # Match results back to candidates
        result_by_id: Dict[str, JudgeResult] = {}
        for jr_dict in llm_results:
            cid = str(jr_dict.get("id", ""))
            overall = jr_dict.get("overall", 5.0)
            fail = jr_dict.get("fail", False)
            tags = jr_dict.get("tags", [])
            if not isinstance(overall, (int, float)):
                overall = 5.0
            overall = max(0.0, min(10.0, float(overall)))
            if not isinstance(fail, bool):
                fail = False
            if not isinstance(tags, list):
                tags = []
            result_by_id[cid] = JudgeResult(overall=overall, fail=fail, tags=tags)

        # Populate cache and results
        for cand, key in zip(uncached, uncached_keys):
            cand_pid = cand.product_id or ""
            jr = result_by_id.get(cand_pid)
            if jr is None:
                jr = JudgeResult(overall=5.0, fail=False, tags=["missing_from_llm"])
            self._cache_put(key, jr)
            results[cand_pid] = jr

        return results

    # ------------------------------------------------------------------
    # LLM call (with retry)
    # ------------------------------------------------------------------

    def _call_llm(self, user_message: str) -> List[Dict]:
        """Call OpenAI and parse the JSON response.
        On failure returns empty list (graceful degradation).
        """
        for attempt in range(1 + self.max_retries):
            try:
                t0 = _time.monotonic()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=2000,
                    timeout=self.timeout,
                )
                elapsed = _time.monotonic() - t0
                content = response.choices[0].message.content or ""
                parsed = self._parse_response(content)
                logger.info(
                    "LLM judge call: %.1fs, %d results parsed",
                    elapsed, len(parsed),
                )
                return parsed
            except Exception as e:
                logger.warning(
                    "LLM judge call failed (attempt %d/%d): %s",
                    attempt + 1, 1 + self.max_retries, e,
                )
                if attempt < self.max_retries:
                    _time.sleep(2.0 * (attempt + 1))

        logger.error("LLM judge: all attempts failed, returning empty results")
        return []

    @staticmethod
    def _parse_response(content: str) -> List[Dict]:
        """Parse LLM JSON response into list of result dicts."""
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            logger.warning("LLM judge: invalid JSON: %.200s", content)
            return []

        if isinstance(data, dict):
            results = data.get("results")
            if isinstance(results, list):
                return results
            # Flat dict with candidate IDs as keys
            if all(isinstance(v, dict) for v in data.values()):
                return [{"id": k, **v} for k, v in data.items()]

        if isinstance(data, list):
            return data

        logger.warning("LLM judge: unexpected structure: %.200s", content)
        return []


# =============================================================================
# 5. SINGLETON
# =============================================================================

_judge: Optional[LLMPairJudge] = None
_judge_lock = threading.Lock()


def get_pair_judge() -> Optional[LLMPairJudge]:
    """Get or create LLMPairJudge singleton. Returns None if disabled or no API key."""
    global _judge
    if _judge is None:
        with _judge_lock:
            if _judge is None:
                try:
                    from config.settings import get_settings
                    settings = get_settings()
                    api_key = settings.openai_api_key
                    if not api_key or not getattr(settings, "llm_judge_enabled", True):
                        logger.info("LLM pair judge disabled (no API key or disabled)")
                        return None
                    model = getattr(settings, "llm_judge_model", "gpt-4.1-mini")
                    timeout = getattr(settings, "llm_judge_timeout", 15.0)
                    _judge = LLMPairJudge(
                        api_key=api_key, model=model, timeout=timeout,
                    )
                except Exception as e:
                    logger.warning("Failed to init LLM pair judge: %s", e)
                    return None
    return _judge
