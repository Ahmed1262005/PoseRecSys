"""
Stylist Ranking Judge for Complete the Fit v3.3
================================================

GPT-4.1 multimodal judge that operates as a **fashion critic**, not a
compatibility filter.  Replaces the v3.0 pass/fail VisionJudge.

Two judging stages:
  1. **Per-category reranking** — ``rerank_category()``:
     Takes source image + top 16 TATTOO candidates.  Returns a full
     ranked ordering from best to worst outfit pairing.

  2. **Outfit composition ranking** — ``rank_outfits()``:
     Takes source image + top 6 complete outfit combos (all categories).
     Returns best-to-worst ranking by overall styling quality.

Key design principles:
  - Images > text.  Attributes lie; images don't.
  - "Would a stylist build this outfit?" not "Can this work?"
  - Penalise catalog-safe/boring pairings, reward intentional styling.
  - Graceful degradation: if LLM fails → keep TATTOO order.

Integration:
  Called from outfit_engine._score_category() and build_outfit() after
  TATTOO scoring.  Reorders candidates instead of removing them.
"""

import hashlib
import json
import logging
import threading
import time as _time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from services.outfit_engine import AestheticProfile

logger = logging.getLogger(__name__)


# =============================================================================
# 0. IMAGE URL VALIDATION
# =============================================================================

_VALID_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp"})
_IMAGE_FORMAT_PARAMS = frozenset({
    "fmt=jpeg", "fmt=jpg", "fmt=png", "fmt=webp", "fmt=gif",
})


def _is_valid_image_url(url: str) -> bool:
    """Check if *url* points to a supported image for the OpenAI vision API.

    Accepts:
      - URLs ending in a known image extension (.jpg, .png, .webp, etc.)
      - Dynamic image CDN URLs with explicit format params (e.g. fmt=jpeg)
    Rejects:
      - Empty / non-HTTP URLs
      - HTML pages (.html, .htm)
      - URLs with no extension and no format hint
    """
    if not url or not url.startswith("http"):
        return False
    parsed = urlparse(url)
    path = parsed.path.lower()
    if path.endswith((".html", ".htm", ".php", ".asp", ".aspx")):
        return False
    for ext in _VALID_IMAGE_EXTENSIONS:
        if path.endswith(ext):
            return True
    query = (parsed.query or "").lower()
    if any(p in query for p in _IMAGE_FORMAT_PARAMS):
        return True
    host = parsed.hostname or ""
    if "s3." in host or "cloudfront" in host:
        return True
    return False


# AVIF magic-bytes detection — AVIF files have "ftyp" at byte offset 4.
# Some retailers (Mango, Garage, Old Navy, Dynamite, Ba&sh) serve AVIF
# images with .jpg extensions and image/jpeg content-type headers.
# OpenAI vision API does NOT support AVIF → instant 400 error.
_AVIF_FTYP = b"ftyp"
_AVIF_BRANDS = frozenset({b"avif", b"avis", b"mif1", b"msf1"})

# Thread-safe cache: url → bool (True = supported, False = AVIF/bad)
_format_cache: Dict[str, bool] = {}
_format_cache_lock = threading.Lock()
_FORMAT_CACHE_SIZE = 512


def _is_supported_image_format(url: str, timeout: float = 3.0) -> bool:
    """Check that the actual image bytes are in a format OpenAI accepts.

    Downloads only the first 32 bytes (HTTP range request) and checks
    for AVIF magic bytes.  Falls back to True (optimistic) on any
    network error — better to let OpenAI reject one image than to
    block all candidates on a slow CDN.

    Results are cached in-memory for the process lifetime.
    """
    if not url:
        return False

    with _format_cache_lock:
        cached = _format_cache.get(url)
    if cached is not None:
        return cached

    supported = True  # optimistic default
    try:
        import requests as _req
        resp = _req.get(
            url, headers={"Range": "bytes=0-31"}, timeout=timeout, stream=True,
        )
        # Accept 200 (full) or 206 (partial)
        if resp.status_code in (200, 206):
            data = resp.content[:32]
            if len(data) >= 12 and data[4:8] == _AVIF_FTYP:
                brand = data[8:12]
                if brand in _AVIF_BRANDS:
                    logger.info("AVIF detected (brand=%s): %s", brand, url[:80])
                    supported = False
    except Exception:
        pass  # network error → optimistic, let OpenAI try

    with _format_cache_lock:
        if len(_format_cache) >= _FORMAT_CACHE_SIZE:
            # Evict ~25% of oldest entries (dict insertion order)
            keys = list(_format_cache.keys())
            for k in keys[:_FORMAT_CACHE_SIZE // 4]:
                _format_cache.pop(k, None)
        _format_cache[url] = supported

    return supported


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
        raw = f"{self.occasion_target}:{self.silhouette_intent}:{self.vibe}:{self.warmth_target}"
        return hashlib.md5(raw.encode()).hexdigest()[:8]

    def context_line(self) -> str:
        return (
            f"Source: {self.occasion_target} {self.source_category.lower()}. "
            f"Building a {self.occasion_target} outfit with {self.vibe} style."
        )


def derive_fit_intent(source: "AestheticProfile") -> FitIntent:
    """Derive outfit intent from a source product's AestheticProfile."""
    occasion_counts: Dict[str, int] = {}
    for occ in (source.occasions or []):
        bucket = _OCCASION_TO_BUCKET.get(occ.lower().strip(), "casual")
        occasion_counts[bucket] = occasion_counts.get(bucket, 0) + 1
    occasion_target = (
        max(occasion_counts, key=lambda k: occasion_counts[k])
        if occasion_counts else "casual"
    )

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

    vibe = (source.primary_style or "casual").lower().strip()
    source_category = source.gemini_category_l2 or source.category or "item"
    warmth_target = source.temp_band or "mild"

    return FitIntent(
        occasion_target=occasion_target,
        silhouette_intent=silhouette_intent,
        vibe=vibe,
        source_category=source_category,
        warmth_target=warmth_target,
    )


# =============================================================================
# 2. PROMPTS — fashion critic, not compatibility checker
# =============================================================================

_RERANK_SYSTEM_PROMPT = (
    "You are a professional fashion stylist building outfits for clients. "
    "You have impeccable taste and high standards. You judge outfit pairings "
    "the way a Net-a-Porter stylist would — not like a search engine.\n\n"
    "Your job: rank candidate garments from BEST to WORST for creating a "
    "cohesive, intentionally styled outfit with the source garment.\n\n"
    "WHAT MAKES A GREAT PAIRING:\n"
    "- Clear outfit story (modern classic, elevated casual, chic weekend, etc.)\n"
    "- Complementary textures/fabrics — knit with denim, silk with structured cotton\n"
    "- Intentional color palette — not everything matching, but harmonious\n"
    "- Matching lifestyle and polish level\n"
    "- Balanced proportions — mix of fitted and relaxed\n"
    "- Would look deliberately styled in a street-style photo\n\n"
    "WHAT MAKES A POOR PAIRING:\n"
    "- Technically compatible but boring/generic (all neutrals, no visual interest)\n"
    "- Wrong lifestyle context (gym pieces with tailored, loungewear with dressy)\n"
    "- Same texture/color as source (matchy-matchy, not curated)\n"
    "- Low fashion credibility — looks accidental, not styled\n"
    "- Sloppy or mismatched vibe, even if individual pieces are fine\n\n"
    "IMPORTANT: Prefer STRONG outfits over SAFE outfits. An interesting pairing "
    "with character beats a generic neutral-on-neutral match every time.\n\n"
    "Return ONLY valid JSON."
)

_OUTFIT_RANKING_SYSTEM_PROMPT = (
    "You are a professional fashion stylist reviewing complete outfit options "
    "for a client. You have the eye of a Vogue editor — you can instantly "
    "tell which outfit tells the best style story.\n\n"
    "WHAT MAKES THE BEST OUTFIT:\n"
    "- Tells a clear style story — you can name the vibe in one phrase\n"
    "- Has intentional contrast: texture, structure, proportion, or color\n"
    "- Feels like a stylist curated it, not an algorithm\n"
    "- Would photograph well as a complete look\n"
    "- Each piece has a role: anchor, complement, or statement\n\n"
    "WHAT MAKES A WEAK OUTFIT:\n"
    "- Everything the same color/texture/vibe (boring, no dimension)\n"
    "- Pieces that don't relate to each other (random assembly)\n"
    "- All safe neutrals with no visual interest\n"
    "- Lifestyle clash between any pieces\n"
    "- Would look like separate items worn together, not a look\n\n"
    "IMPORTANT: A bold, well-styled outfit always beats a safe, generic one.\n\n"
    "Return ONLY valid JSON."
)


# =============================================================================
# 3. MESSAGE BUILDERS
# =============================================================================

def build_rerank_messages(
    source: "AestheticProfile",
    candidates: List["AestheticProfile"],
    intent: FitIntent,
    detail: str = "auto",
) -> List[Dict[str, Any]]:
    """Build messages for per-category stylist reranking."""
    content_parts: List[Dict[str, Any]] = []

    # 1. Context + source image
    content_parts.append({
        "type": "text",
        "text": (
            f"I need you to style a {intent.occasion_target} outfit.\n\n"
            f"SOURCE GARMENT (client already owns this — "
            f"{intent.source_category.lower()}, {intent.vibe} style):"
        ),
    })
    source_url = source.image_url or ""
    if _is_valid_image_url(source_url):
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": source_url, "detail": detail},
        })

    # 2. Candidate images with IDs
    content_parts.append({
        "type": "text",
        "text": (
            "\nCANDIDATE PIECES to pair with the source "
            f"({len(candidates)} options):"
        ),
    })
    for i, cand in enumerate(candidates):
        cand_url = cand.image_url or ""
        cand_id = cand.product_id or f"cand_{i}"
        content_parts.append({
            "type": "text",
            "text": f"\n{cand_id}:",
        })
        if _is_valid_image_url(cand_url):
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": cand_url, "detail": detail},
            })

    # 3. Instruction
    content_parts.append({
        "type": "text",
        "text": (
            "\n\nRank ALL candidates from BEST to WORST for building a "
            f"stylish {intent.occasion_target} outfit with the source.\n"
            "Consider: lifestyle match, outfit polish, silhouette balance, "
            "color harmony, fabric coherence, and whether the pairing feels "
            "intentional.\n\n"
            'Return JSON: {"ranking": ["best_id", "second_id", ...], '
            '"note": "one sentence on why the top pick works"}'
        ),
    })

    return [
        {"role": "system", "content": _RERANK_SYSTEM_PROMPT},
        {"role": "user", "content": content_parts},
    ]


def build_outfit_ranking_messages(
    source: "AestheticProfile",
    outfits: List[Dict[str, "AestheticProfile"]],
    intent: FitIntent,
    detail: str = "auto",
) -> List[Dict[str, Any]]:
    """Build messages for complete outfit composition ranking."""
    content_parts: List[Dict[str, Any]] = []

    # 1. Context + source image
    content_parts.append({
        "type": "text",
        "text": (
            f"Review these complete outfit options for a "
            f"{intent.occasion_target} look.\n\n"
            "SOURCE GARMENT (client's piece):"
        ),
    })
    source_url = source.image_url or ""
    if _is_valid_image_url(source_url):
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": source_url, "detail": detail},
        })

    # 2. Each outfit option
    content_parts.append({
        "type": "text",
        "text": "\nCOMPLETE OUTFIT OPTIONS:",
    })
    for i, outfit in enumerate(outfits, 1):
        content_parts.append({
            "type": "text",
            "text": f"\n--- OUTFIT {i} ---",
        })
        for cat, profile in outfit.items():
            img_url = profile.image_url or ""
            if _is_valid_image_url(img_url):
                content_parts.append({
                    "type": "text",
                    "text": f"{cat}:",
                })
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": img_url, "detail": detail},
                })

    # 3. Instruction
    content_parts.append({
        "type": "text",
        "text": (
            f"\nRank these {len(outfits)} outfits from BEST to WORST styled "
            "with the source garment. The best outfit tells a clear style "
            "story with intentional contrast — not just safe matching.\n\n"
            'Return JSON: {"ranking": [best_idx, ..., worst_idx], '
            '"note": "why the winner works"}\n'
            "(indices are 1-based)"
        ),
    })

    return [
        {"role": "system", "content": _OUTFIT_RANKING_SYSTEM_PROMPT},
        {"role": "user", "content": content_parts},
    ]


# --- Legacy message builders (kept for backward compatibility in tests) ------

def build_vision_messages(
    source: "AestheticProfile",
    candidates: List["AestheticProfile"],
    intent: FitIntent,
) -> List[Dict[str, Any]]:
    """Legacy: build pass/fail veto messages. Delegates to rerank builder."""
    return build_rerank_messages(source, candidates, intent, detail="low")


# =============================================================================
# 4. STYLIST JUDGE
# =============================================================================

class StylistJudge:
    """Post-TATTOO stylist ranking judge for Complete the Fit v3.3.

    Replaces the v3.0 VisionJudge pass/fail filter with a full
    ranking-based approach.  Uses GPT-4.1 multimodal to reorder
    candidates by outfit styling quality.

    Singleton — accessed via ``get_pair_judge()``.
    Bounded in-memory LRU cache for ranking results.
    Graceful degradation: if LLM fails → preserve TATTOO order.
    """

    _CACHE_SIZE = 256

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1",
        timeout: float = 30.0,
        max_retries: int = 1,
        detail: str = "auto",
    ):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.detail = detail

        # Cache stores ranking lists keyed on source+candidates+intent
        self._cache: Dict[str, List[str]] = {}
        self._cache_order: List[str] = []
        self._cache_lock = threading.Lock()

        # Last notes from judge calls (read by engine for scoring_info)
        self.last_category_notes: List[str] = []
        self.last_outfit_note: str = ""

        logger.info(
            "StylistJudge initialized: model=%s, detail=%s, timeout=%.1fs",
            model, detail, timeout,
        )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_ranking_cache_key(
        source_pid: str, cand_pids: List[str], intent_hash: str,
    ) -> str:
        cand_str = ",".join(sorted(cand_pids))
        raw = f"{source_pid}:{cand_str}:{intent_hash}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _cache_get(self, key: str) -> Optional[List[str]]:
        with self._cache_lock:
            return self._cache.get(key)

    def _cache_put(self, key: str, value: List[str]) -> None:
        with self._cache_lock:
            if key in self._cache:
                return
            if len(self._cache) >= self._CACHE_SIZE:
                evict = self._cache_order.pop(0)
                self._cache.pop(evict, None)
            self._cache[key] = value
            self._cache_order.append(key)

    # ------------------------------------------------------------------
    # Per-category reranking
    # ------------------------------------------------------------------

    def rerank_category(
        self,
        source: "AestheticProfile",
        candidates: List["AestheticProfile"],
        intent: FitIntent,
    ) -> Optional[List[str]]:
        """Rank candidates by outfit styling quality with the source.

        Args:
            source: the source product
            candidates: top-N TATTOO candidates (with images)
            intent: derived fit intent

        Returns:
            Ordered list of product IDs (best first),
            or None if ranking fails (preserve TATTOO order).
        """
        if not candidates:
            return None

        # Pre-filter: skip if source image is invalid or unsupported format
        source_url = source.image_url or ""
        if not _is_valid_image_url(source_url):
            logger.warning(
                "Stylist rerank: source %s has invalid image URL, skipping",
                source.product_id,
            )
            return None
        if not _is_supported_image_format(source_url):
            logger.warning(
                "Stylist rerank: source %s has unsupported format (AVIF?), skipping",
                source.product_id,
            )
            return None

        # Pre-filter: only send candidates with valid, supported image URLs
        valid = [
            c for c in candidates
            if _is_valid_image_url(c.image_url or "")
            and _is_supported_image_format(c.image_url or "")
        ]
        skipped = len(candidates) - len(valid)
        if skipped:
            logger.info(
                "Stylist rerank: filtered %d/%d candidates (invalid/AVIF URLs)",
                skipped, len(candidates),
            )
        if not valid:
            return None

        source_pid = source.product_id or ""
        cand_pids = [c.product_id or "" for c in valid]
        intent_hash = intent.cache_key()

        # Check cache
        cache_key = self._make_ranking_cache_key(source_pid, cand_pids, intent_hash)
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.debug("Stylist rerank: cache hit for source %s", source_pid)
            return cached

        logger.info(
            "Stylist rerank: %d candidates for source %s (%s %s)",
            len(valid), source_pid, intent.occasion_target, intent.vibe,
        )

        # Build messages and call LLM
        messages = build_rerank_messages(source, valid, intent, self.detail)
        ranking = self._call_rerank(messages, cand_pids)

        if ranking is not None:
            self._cache_put(cache_key, ranking)

        return ranking

    def _call_rerank(
        self, messages: List[Dict], cand_pids: List[str],
    ) -> Optional[List[str]]:
        """Call LLM and parse category ranking response.

        Returns ordered list of product IDs, or None on failure.
        """
        for attempt in range(1 + self.max_retries):
            try:
                t0 = _time.monotonic()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=800,
                    timeout=self.timeout,
                )
                elapsed = _time.monotonic() - t0
                content = response.choices[0].message.content or ""
                ranking = self._parse_rerank_response(content, cand_pids)
                if ranking:
                    note = ""
                    try:
                        note = json.loads(content).get("note", "")
                    except Exception:
                        pass
                    self.last_category_notes.append(note)
                    logger.info(
                        "Stylist rerank: %.1fs, top=%s, note=%s",
                        elapsed, ranking[0][:12] if ranking else "?",
                        note[:80],
                    )
                return ranking
            except Exception as e:
                logger.warning(
                    "Stylist rerank failed (attempt %d/%d): %s",
                    attempt + 1, 1 + self.max_retries, e,
                )
                if attempt < self.max_retries:
                    _time.sleep(1.5 * (attempt + 1))

        logger.error("Stylist rerank: all attempts failed, keeping TATTOO order")
        return None

    @staticmethod
    def _parse_rerank_response(
        content: str, cand_pids: List[str],
    ) -> Optional[List[str]]:
        """Parse LLM ranking response into ordered list of product IDs."""
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Stylist rerank: invalid JSON: %.200s", content)
            return None

        if not isinstance(data, dict):
            logger.warning("Stylist rerank: not a dict: %.200s", content)
            return None

        ranking_raw = data.get("ranking", [])
        if not isinstance(ranking_raw, list) or not ranking_raw:
            logger.warning("Stylist rerank: no ranking list: %.200s", content)
            return None

        # Build ordered list, matching IDs to known candidates
        pid_set = set(cand_pids)
        ordered: List[str] = []
        seen: Set[str] = set()
        for item in ranking_raw:
            pid = str(item).strip()
            if pid in pid_set and pid not in seen:
                ordered.append(pid)
                seen.add(pid)

        if not ordered:
            logger.warning("Stylist rerank: no matching IDs in ranking")
            return None

        # Append any candidates not mentioned (preserve TATTOO order for those)
        for pid in cand_pids:
            if pid not in seen:
                ordered.append(pid)

        return ordered

    # ------------------------------------------------------------------
    # Outfit-level composition ranking
    # ------------------------------------------------------------------

    def rank_outfits(
        self,
        source: "AestheticProfile",
        outfits: List[Dict[str, "AestheticProfile"]],
        intent: FitIntent,
    ) -> Optional[List[int]]:
        """Rank complete outfit combinations by styling quality.

        Returns:
            0-indexed list of outfit indices ordered best->worst,
            or None if ranking fails (preserve TATTOO order).
        """
        if len(outfits) <= 1:
            return None

        source_url = source.image_url or ""
        if not _is_valid_image_url(source_url) or not _is_supported_image_format(source_url):
            logger.warning(
                "Outfit ranking: source %s has invalid/unsupported image, skipping",
                source.product_id,
            )
            return None

        # Pre-filter outfits with invalid or unsupported-format image URLs
        def _outfit_images_ok(outfit: Dict[str, "AestheticProfile"]) -> bool:
            return all(
                _is_valid_image_url(p.image_url or "")
                and _is_supported_image_format(p.image_url or "")
                for p in outfit.values()
            )

        valid_outfits: List[Dict[str, "AestheticProfile"]] = []
        valid_indices: List[int] = []
        for i, outfit in enumerate(outfits):
            if _outfit_images_ok(outfit):
                valid_outfits.append(outfit)
                valid_indices.append(i)

        if len(valid_outfits) <= 1:
            return None

        logger.info(
            "Outfit ranking: %d combos for source %s",
            len(valid_outfits), source.product_id,
        )

        messages = build_outfit_ranking_messages(
            source, valid_outfits, intent, self.detail,
        )

        for attempt in range(1 + self.max_retries):
            try:
                t0 = _time.monotonic()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=300,
                    timeout=self.timeout,
                )
                elapsed = _time.monotonic() - t0
                content = response.choices[0].message.content or ""
                ranking = self._parse_outfit_ranking(content, len(valid_outfits))
                if ranking is not None:
                    # Remap back to original indices
                    remapped = [valid_indices[r] for r in ranking]
                    excluded = [
                        i for i in range(len(outfits))
                        if i not in set(valid_indices)
                    ]
                    ranking = remapped + excluded
                note = ""
                try:
                    note = json.loads(content).get("note", "")
                except Exception:
                    pass
                self.last_outfit_note = note
                logger.info(
                    "Outfit ranking: %.1fs, result=%s, note=%s",
                    elapsed, ranking, note[:80],
                )
                return ranking
            except Exception as e:
                logger.warning(
                    "Outfit ranking failed (attempt %d/%d): %s",
                    attempt + 1, 1 + self.max_retries, e,
                )
                if attempt < self.max_retries:
                    _time.sleep(1.5 * (attempt + 1))

        logger.error("Outfit ranking: all attempts failed, keeping TATTOO order")
        return None

    @staticmethod
    def _parse_outfit_ranking(
        content: str, n_outfits: int,
    ) -> Optional[List[int]]:
        """Parse outfit ranking JSON into 0-indexed list."""
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Outfit ranking: invalid JSON: %.200s", content)
            return None

        if not isinstance(data, dict):
            return None

        ranking = data.get("ranking", [])
        if not isinstance(ranking, list) or not ranking:
            return None

        # Convert 1-indexed to 0-indexed
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
            return None

        # Fill omitted indices at end
        for i in range(n_outfits):
            if i not in seen:
                indices.append(i)

        return indices

    # ------------------------------------------------------------------
    # Legacy API compatibility
    # ------------------------------------------------------------------

    def judge_batch(
        self,
        source: "AestheticProfile",
        candidates: List["AestheticProfile"],
        intent: FitIntent,
    ) -> Set[str]:
        """Legacy pass/fail API — converts ranking to bottom-3 fails.

        Kept for backward compatibility with existing tests and code
        that hasn't migrated to rerank_category() yet.
        """
        ranking = self.rerank_category(source, candidates, intent)
        if not ranking or len(ranking) <= 3:
            return set()
        # Bottom 30% are considered "fails" for legacy callers
        cutoff = max(len(ranking) - 3, len(ranking) * 7 // 10)
        return set(ranking[cutoff:])


# Legacy alias
VisionJudge = StylistJudge


# =============================================================================
# 5. SINGLETON
# =============================================================================

_judge: Optional[StylistJudge] = None
_judge_lock = threading.Lock()


def get_pair_judge() -> Optional[StylistJudge]:
    """Get or create StylistJudge singleton.

    Returns None if disabled or no API key.
    """
    global _judge
    if _judge is None:
        with _judge_lock:
            if _judge is None:
                try:
                    from config.settings import get_settings
                    settings = get_settings()
                    api_key = settings.openai_api_key
                    if not api_key or not getattr(settings, "llm_judge_enabled", True):
                        logger.info("Stylist judge disabled (no API key or disabled)")
                        return None
                    model = getattr(settings, "llm_judge_model", "gpt-4.1")
                    timeout = getattr(settings, "llm_judge_timeout", 30.0)
                    detail = getattr(settings, "llm_judge_detail", "auto")
                    _judge = StylistJudge(
                        api_key=api_key,
                        model=model,
                        timeout=timeout,
                        detail=detail,
                    )
                except Exception as e:
                    logger.warning("Failed to init stylist judge: %s", e)
                    return None
    return _judge
