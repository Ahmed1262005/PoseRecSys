"""
Vision-Based Pass/Fail Judge for Complete the Fit v3
=====================================================

Sends source + candidate product images to gpt-4o-mini vision and asks
which candidates visually CLASH with the source for the given occasion.
Operates as a post-TATTOO filter: TATTOO ranks, vision judge vetoes.

Components:
  - FitIntent: derived from source product, provides minimal text context
  - derive_fit_intent(): source AestheticProfile -> FitIntent
  - VisionJudge: singleton OpenAI client with bounded LRU cache
  - get_pair_judge(): thread-safe singleton accessor

Integration:
  Called from outfit_engine._score_category() after TATTOO scoring.
  Returns set of product IDs that should be removed (visual clashes).
  No score blending — TATTOO ranking is preserved for passing candidates.
"""

import hashlib
import json
import logging
import threading
import time as _time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from services.outfit_engine import AestheticProfile

logger = logging.getLogger(__name__)


# =============================================================================
# 1. FIT INTENT  (derived from source product — minimal context for prompt)
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
    source_category: str            # e.g. "Jogger", "Blouse"
    warmth_target: str              # hot | mild | cold | any

    def cache_key(self) -> str:
        """Short hash for cache keying."""
        raw = f"{self.occasion_target}:{self.silhouette_intent}:{self.vibe}:{self.warmth_target}"
        return hashlib.md5(raw.encode()).hexdigest()[:8]

    def context_line(self) -> str:
        """One-line text context for the vision prompt."""
        return (
            f"Source: {self.occasion_target} {self.source_category.lower()}. "
            f"Building a {self.occasion_target} outfit with {self.vibe} style."
        )


def derive_fit_intent(source: "AestheticProfile") -> FitIntent:
    """Derive outfit intent from a source product's AestheticProfile."""
    # Occasion target
    occasion_counts: Dict[str, int] = {}
    for occ in (source.occasions or []):
        bucket = _OCCASION_TO_BUCKET.get(occ.lower().strip(), "casual")
        occasion_counts[bucket] = occasion_counts.get(bucket, 0) + 1
    occasion_target = max(occasion_counts, key=lambda k: occasion_counts[k]) if occasion_counts else "casual"

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

    # Source category
    source_category = source.gemini_category_l2 or source.category or "item"

    # Warmth target
    warmth_target = source.temp_band or "mild"

    return FitIntent(
        occasion_target=occasion_target,
        silhouette_intent=silhouette_intent,
        vibe=vibe,
        source_category=source_category,
        warmth_target=warmth_target,
    )


# =============================================================================
# 2. PROMPT
# =============================================================================

_SYSTEM_PROMPT = (
    "You are a fashion outfit visual compatibility judge. "
    "You will see a SOURCE garment image followed by CANDIDATE garment images. "
    "Your job: identify which candidates would visually CLASH with the source "
    "when worn together as an outfit.\n\n"
    "CLASH means:\n"
    "- Colors that fight each other (e.g. competing neon brights)\n"
    "- Wildly mismatched formality (gym shorts with a silk blouse)\n"
    "- Clashing patterns (busy print + busy print)\n"
    "- Same exact look as the source (redundant, not complementary)\n"
    "- Textures/fabrics that look wrong together\n\n"
    "Most candidates should PASS. Only flag clear visual clashes.\n"
    "When in doubt, PASS. We want to remove bad matches, not be picky.\n\n"
    "Return ONLY valid JSON: {\"fail\": [\"id_of_clashing_candidate\", ...]}\n"
    "If all candidates pass, return: {\"fail\": []}"
)

_OUTFIT_RANKING_PROMPT = (
    "You are a fashion stylist ranking complete outfit combinations. "
    "You will see a SOURCE garment, then several OUTFIT OPTIONS. "
    "Each option shows the garments that would complete the outfit.\n\n"
    "Rank from BEST to WORST. A great outfit has:\n"
    "- Color story: intentional palette, not everything matching\n"
    "- Texture contrast: e.g. knit + denim, silk + leather, linen + structured cotton\n"
    "- Balanced proportions: mix of fitted and relaxed\n"
    "- Style intent: looks deliberately styled, not randomly safe\n\n"
    "A mediocre outfit:\n"
    "- Everything the same color/texture (boring)\n"
    "- All safe neutrals with no visual interest\n"
    "- Looks like separate items, not a curated outfit\n\n"
    "Return ONLY valid JSON: {\"ranking\": [3, 1, 2, ...]}\n"
    "Numbers are outfit indices (1-based), best to worst. Rank ALL outfits."
)


def build_vision_messages(
    source: "AestheticProfile",
    candidates: List["AestheticProfile"],
    intent: FitIntent,
) -> List[Dict[str, Any]]:
    """Build the OpenAI messages array with images for the vision judge."""
    # User message content parts: text context + source image + candidate images
    content_parts: List[Dict[str, Any]] = []

    # 1. Text context line
    content_parts.append({
        "type": "text",
        "text": intent.context_line() + "\n\nSOURCE garment:",
    })

    # 2. Source image
    source_url = source.image_url or ""
    if source_url:
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": source_url, "detail": "low"},
        })

    # 3. Candidate images with IDs
    content_parts.append({
        "type": "text",
        "text": "\nCANDIDATE garments to evaluate:",
    })

    for i, cand in enumerate(candidates):
        cand_url = cand.image_url or ""
        cand_id = cand.product_id or f"cand_{i}"
        content_parts.append({
            "type": "text",
            "text": f"\nCandidate {cand_id}:",
        })
        if cand_url:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": cand_url, "detail": "low"},
            })

    # 4. Final instruction
    content_parts.append({
        "type": "text",
        "text": (
            "\n\nWhich candidates visually CLASH with the source for "
            f"a {intent.occasion_target} outfit? Return JSON: "
            '{"fail": ["id1", "id2", ...]} or {"fail": []} if all pass.'
        ),
    })

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": content_parts},
    ]


def build_outfit_ranking_messages(
    source: "AestheticProfile",
    outfits: List[Dict[str, "AestheticProfile"]],
    intent: FitIntent,
) -> List[Dict[str, Any]]:
    """Build OpenAI messages to rank complete outfit combinations.

    Each outfit is a dict mapping category -> AestheticProfile.
    The judge sees the source image + all pieces for each outfit option.
    """
    content_parts: List[Dict[str, Any]] = []

    # 1. Context + source image
    content_parts.append({
        "type": "text",
        "text": intent.context_line() + "\n\nSOURCE garment (already owned):",
    })
    source_url = source.image_url or ""
    if source_url:
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": source_url, "detail": "low"},
        })

    # 2. Each outfit option
    content_parts.append({
        "type": "text",
        "text": "\nHere are the outfit options to complete the look:",
    })
    for i, outfit in enumerate(outfits, 1):
        content_parts.append({
            "type": "text",
            "text": f"\n--- OUTFIT {i} ---",
        })
        for cat, profile in outfit.items():
            img_url = profile.image_url or ""
            if img_url:
                content_parts.append({
                    "type": "text",
                    "text": f"{cat}:",
                })
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": img_url, "detail": "low"},
                })

    # 3. Final instruction
    content_parts.append({
        "type": "text",
        "text": (
            f"\nRank these {len(outfits)} outfits from BEST to WORST styled "
            "with the source garment. Prefer intentional contrast and styling "
            "over safe matching. Return JSON: "
            '{"ranking": [best_idx, ..., worst_idx]} (1-indexed)'
        ),
    })

    return [
        {"role": "system", "content": _OUTFIT_RANKING_PROMPT},
        {"role": "user", "content": content_parts},
    ]


# =============================================================================
# 3. VISION JUDGE
# =============================================================================

class VisionJudge:
    """Post-TATTOO vision filter that vetoes visually clashing candidates.

    Singleton — accessed via get_pair_judge().
    Uses gpt-4o-mini with image inputs.
    Bounded in-memory LRU cache for results.
    """

    _CACHE_SIZE = 256

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout: float = 20.0,
        max_retries: int = 1,
    ):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        self._cache: Dict[str, bool] = {}     # key -> True means FAIL
        self._cache_order: List[str] = []
        self._cache_lock = threading.Lock()

        logger.info(
            "VisionJudge initialized: model=%s, timeout=%.1fs, cache_size=%d",
            model, timeout, self._CACHE_SIZE,
        )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_cache_key(source_pid: str, cand_pid: str, intent_hash: str) -> str:
        return f"{source_pid}:{cand_pid}:{intent_hash}"

    def _cache_get(self, key: str) -> Optional[bool]:
        with self._cache_lock:
            return self._cache.get(key)

    def _cache_put(self, key: str, value: bool) -> None:
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
    ) -> Set[str]:
        """Evaluate candidates visually against source.

        Returns set of product IDs that FAIL (should be removed).
        On LLM failure, returns empty set (graceful degradation = keep all).
        """
        if not candidates:
            return set()

        source_pid = source.product_id or ""
        intent_hash = intent.cache_key()

        # Separate cached from uncached
        fail_ids: Set[str] = set()
        uncached: List["AestheticProfile"] = []

        for cand in candidates:
            cand_pid = cand.product_id or ""
            key = self._make_cache_key(source_pid, cand_pid, intent_hash)
            cached = self._cache_get(key)
            if cached is not None:
                if cached:  # True = fail
                    fail_ids.add(cand_pid)
            else:
                uncached.append(cand)

        if not uncached:
            logger.debug("Vision judge: all %d candidates cached", len(candidates))
            return fail_ids

        logger.info(
            "Vision judge: %d cached, %d to evaluate for source %s",
            len(candidates) - len(uncached), len(uncached), source_pid,
        )

        # Build messages and call LLM
        messages = build_vision_messages(source, uncached, intent)
        new_fails = self._call_vision(messages)

        # Populate cache
        for cand in uncached:
            cand_pid = cand.product_id or ""
            key = self._make_cache_key(source_pid, cand_pid, intent_hash)
            is_fail = cand_pid in new_fails
            self._cache_put(key, is_fail)
            if is_fail:
                fail_ids.add(cand_pid)

        return fail_ids

    # ------------------------------------------------------------------
    # Vision call (with retry)
    # ------------------------------------------------------------------

    def _call_vision(self, messages: List[Dict]) -> Set[str]:
        """Call OpenAI vision API and parse the fail list.
        On failure returns empty set (graceful degradation = keep all).
        """
        for attempt in range(1 + self.max_retries):
            try:
                t0 = _time.monotonic()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=500,
                    timeout=self.timeout,
                )
                elapsed = _time.monotonic() - t0
                content = response.choices[0].message.content or ""
                fails = self._parse_response(content)
                logger.info(
                    "Vision judge call: %.1fs, %d fails",
                    elapsed, len(fails),
                )
                return fails
            except Exception as e:
                logger.warning(
                    "Vision judge call failed (attempt %d/%d): %s",
                    attempt + 1, 1 + self.max_retries, e,
                )
                if attempt < self.max_retries:
                    _time.sleep(2.0 * (attempt + 1))

        logger.error("Vision judge: all attempts failed, keeping all candidates")
        return set()

    @staticmethod
    def _parse_response(content: str) -> Set[str]:
        """Parse LLM JSON response into set of fail IDs."""
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Vision judge: invalid JSON: %.200s", content)
            return set()

        if isinstance(data, dict):
            fail_list = data.get("fail", [])
            if isinstance(fail_list, list):
                return {str(x) for x in fail_list if x}

        logger.warning("Vision judge: unexpected structure: %.200s", content)
        return set()

    # ------------------------------------------------------------------
    # Outfit-level ranking
    # ------------------------------------------------------------------

    def rank_outfits(
        self,
        source: "AestheticProfile",
        outfits: List[Dict[str, "AestheticProfile"]],
        intent: FitIntent,
    ) -> Optional[List[int]]:
        """Rank complete outfit combinations by visual styling quality.

        Args:
            source: the source product (already owned)
            outfits: list of outfit combos, each mapping category -> AestheticProfile
            intent: derived fit intent for prompt context

        Returns:
            0-indexed list of outfit indices ordered best->worst,
            or None if ranking fails (preserve TATTOO order).
        """
        if len(outfits) <= 1:
            return None

        messages = build_outfit_ranking_messages(source, outfits, intent)

        for attempt in range(1 + self.max_retries):
            try:
                t0 = _time.monotonic()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=200,
                    timeout=self.timeout,
                )
                elapsed = _time.monotonic() - t0
                content = response.choices[0].message.content or ""
                ranking = self._parse_ranking_response(content, len(outfits))
                logger.info(
                    "Outfit ranking: %.1fs, result=%s", elapsed, ranking,
                )
                return ranking
            except Exception as e:
                logger.warning(
                    "Outfit ranking failed (attempt %d/%d): %s",
                    attempt + 1, 1 + self.max_retries, e,
                )
                if attempt < self.max_retries:
                    _time.sleep(2.0 * (attempt + 1))

        logger.error("Outfit ranking: all attempts failed, keeping TATTOO order")
        return None

    @staticmethod
    def _parse_ranking_response(content: str, n_outfits: int) -> Optional[List[int]]:
        """Parse ranking JSON into 0-indexed list of outfit indices."""
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Outfit ranking: invalid JSON: %.200s", content)
            return None

        if not isinstance(data, dict):
            logger.warning("Outfit ranking: not a dict: %.200s", content)
            return None

        ranking = data.get("ranking", [])
        if not isinstance(ranking, list):
            logger.warning("Outfit ranking: 'ranking' not a list: %.200s", content)
            return None

        # Convert 1-indexed to 0-indexed, validate
        indices: List[int] = []
        seen: Set[int] = set()
        for r in ranking:
            try:
                idx = int(r) - 1
                if 0 <= idx < n_outfits and idx not in seen:
                    indices.append(idx)
                    seen.add(idx)
            except (ValueError, TypeError):
                continue

        if not indices:
            logger.warning("Outfit ranking: no valid indices parsed")
            return None

        # Fill in any omitted indices at the end (preserve their order)
        for i in range(n_outfits):
            if i not in seen:
                indices.append(i)

        return indices


# =============================================================================
# 4. SINGLETON
# =============================================================================

_judge: Optional[VisionJudge] = None
_judge_lock = threading.Lock()


def get_pair_judge() -> Optional[VisionJudge]:
    """Get or create VisionJudge singleton. Returns None if disabled or no API key."""
    global _judge
    if _judge is None:
        with _judge_lock:
            if _judge is None:
                try:
                    from config.settings import get_settings
                    settings = get_settings()
                    api_key = settings.openai_api_key
                    if not api_key or not getattr(settings, "llm_judge_enabled", True):
                        logger.info("Vision judge disabled (no API key or disabled)")
                        return None
                    model = getattr(settings, "llm_judge_model", "gpt-4o-mini")
                    timeout = getattr(settings, "llm_judge_timeout", 20.0)
                    _judge = VisionJudge(
                        api_key=api_key, model=model, timeout=timeout,
                    )
                except Exception as e:
                    logger.warning("Failed to init vision judge: %s", e)
                    return None
    return _judge
