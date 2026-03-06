"""
Gemini Vision Reranker — LLM-based verification for detail-specific queries.

When the query's key distinguishing feature is a non-filterable product detail
(e.g., "zipped pockets", "pearl buttons", "ruched sides"), embeddings and
keyword search can't reliably find matching products.  This module sends
candidate product images to Gemini 2.0 Flash, which scores each image 0-10
for the specific detail.  Scores are used to reorder candidates before
profile scoring and diversity caps.

Architecture:
    RRF merge → [Gemini Vision Reranker] → Profile scoring → Diversity

Only triggered when the planner identifies non-filterable `detail_terms`.
Gracefully degrades: on any failure, returns candidates unchanged.

Cost: ~$0.001/search at 20 images (Gemini 2.0 Flash pricing).
Latency: ~3-6s for 20 images (parallel download + single Gemini call).
"""

import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

from config.settings import get_settings
from core.logging import get_logger

logger = get_logger(__name__)

# Image prep constants
_IMG_SIZE = (256, 256)
_JPEG_QUALITY = 50
_DOWNLOAD_TIMEOUT = 4  # seconds per image
_DOWNLOAD_WORKERS = 8  # parallel downloads


# =============================================================================
# Score blending
# =============================================================================

# Multiplicative factor applied to rrf_score based on LLM relevance.
# Maps LLM score (0-10) to a multiplier via piecewise linear:
#   0-2  → 0.3  (strong demotion — detail clearly absent)
#   3-4  → 0.7  (mild demotion)
#   5-6  → 1.0  (neutral — possible but unconfirmed)
#   7    → 1.3  (mild boost)
#   8-10 → 2.0  (strong boost — detail clearly visible)
def _llm_score_to_factor(score: int) -> float:
    if score <= 2:
        return 0.3
    elif score <= 4:
        return 0.7
    elif score <= 6:
        return 1.0
    elif score == 7:
        return 1.3
    else:
        return 2.0


# =============================================================================
# Image preparation
# =============================================================================

def _download_and_convert(url: str) -> Optional[bytes]:
    """Download an image URL and convert to small JPEG bytes.

    Handles JPEG, PNG, WebP, AVIF, and any format PIL can open.
    Returns None on any failure (network, format, corruption).
    """
    try:
        resp = requests.get(url, timeout=_DOWNLOAD_TIMEOUT)
        if resp.status_code != 200 or len(resp.content) < 500:
            return None
        img = Image.open(io.BytesIO(resp.content))
        buf = io.BytesIO()
        img.convert("RGB").resize(_IMG_SIZE).save(
            buf, format="JPEG", quality=_JPEG_QUALITY,
        )
        return buf.getvalue()
    except Exception:
        return None


def _download_images_parallel(
    candidates: List[dict],
) -> List[Tuple[int, dict, bytes]]:
    """Download images for candidates in parallel.

    Returns list of (index, candidate_dict, jpeg_bytes) for successful
    downloads only.  Skips candidates without image_url or failed downloads.
    """
    tasks: List[Tuple[int, dict, str]] = []
    for i, c in enumerate(candidates):
        url = c.get("image_url")
        if url:
            tasks.append((i, c, url))

    results: List[Tuple[int, dict, bytes]] = []
    with ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as pool:
        future_map = {
            pool.submit(_download_and_convert, url): (idx, cand)
            for idx, cand, url in tasks
        }
        for future in as_completed(future_map):
            idx, cand = future_map[future]
            try:
                jpeg = future.result()
                if jpeg:
                    results.append((idx, cand, jpeg))
            except Exception:
                pass

    # Sort by original index to maintain order
    results.sort(key=lambda x: x[0])
    return results


# =============================================================================
# Gemini API call
# =============================================================================

def _build_prompt(detail: str, items: List[Tuple[int, dict]]) -> str:
    """Build the scoring prompt text."""
    product_list = "\n".join(
        f'{i}. [{c.get("brand", "?")}] {(c.get("name") or "?")[:60]}'
        for i, (_, c) in enumerate(items, 1)
    )
    return f"""Score each product image 0-10 for the specific detail: "{detail}"

Scoring guide:
- 8-10: The detail "{detail}" is clearly visible in the image
- 5-7: Likely has it based on garment style, or partially visible
- 3-4: Uncertain — pockets/features exist but detail not confirmed
- 0-2: Detail is clearly absent or not visible at all

IMPORTANT: Be precise. A jacket with a front zipper closure is NOT the same as
a jacket with zippered pockets. A dress with side gathering is NOT the same as
ruched sides. Score based on the SPECIFIC detail requested.

Return ONLY a JSON array: [{{"idx":1,"score":N,"reason":"brief"}}]

Products:
{product_list}"""


def _call_gemini(
    prompt: str,
    images: List[bytes],
    items: List[Tuple[int, dict]],
    model: str = "gemini-2.0-flash",
    api_key: str = "",
) -> Dict[int, Tuple[int, str]]:
    """Call Gemini with images and parse scores.

    Args:
        prompt: The scoring prompt text.
        images: List of JPEG bytes, one per item.
        items: List of (original_index, candidate_dict) matching images.
        model: Gemini model name.
        api_key: Google API key.

    Returns:
        Dict mapping original_candidate_index -> (score, reason).
        Empty dict on failure.
    """
    try:
        from google import genai
    except ImportError:
        logger.warning("google-genai package not installed, skipping vision reranker")
        return {}

    if not api_key:
        logger.warning("No GOOGLE_API_KEY configured, skipping vision reranker")
        return {}

    try:
        client = genai.Client(api_key=api_key)

        # Build multimodal content: prompt + interleaved (label, image) pairs
        parts: list = [prompt + "\n"]
        for i, ((orig_idx, cand), jpeg) in enumerate(zip(items, images), 1):
            parts.append(f'{i}. [{cand.get("brand", "?")}] {(cand.get("name") or "?")[:60]}')
            parts.append(
                genai.types.Part.from_bytes(data=jpeg, mime_type="image/jpeg")
            )

        resp = client.models.generate_content(
            model=model,
            contents=parts,
            config=genai.types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1200,
            ),
        )

        # Parse JSON array from response
        text = resp.text or ""
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            logger.warning(
                "Vision reranker: could not parse JSON from response",
                response_preview=text[:200],
            )
            return {}

        scores_list = json.loads(match.group())
        result: Dict[int, Tuple[int, str]] = {}
        for entry in scores_list:
            llm_idx = entry.get("idx", 0) - 1  # 1-indexed in prompt
            score = int(entry.get("score", 0))
            reason = str(entry.get("reason", ""))
            if 0 <= llm_idx < len(items):
                orig_idx = items[llm_idx][0]
                result[orig_idx] = (score, reason)

        return result

    except Exception as e:
        logger.warning(
            "Vision reranker Gemini call failed",
            error=str(e),
            model=model,
        )
        return {}


# =============================================================================
# Main reranker entry point
# =============================================================================

def rerank_with_vision(
    detail_terms: List[str],
    candidates: List[dict],
    max_candidates: int = 20,
) -> List[dict]:
    """Rerank candidates using Gemini vision scoring for specific details.

    Downloads product images, sends them to Gemini 2.0 Flash for scoring,
    and adjusts rrf_score based on how well each product matches the
    requested detail.

    Args:
        detail_terms: Non-filterable product details from the planner
                      (e.g., ["zipped pockets"]).
        candidates: RRF-merged candidate dicts with image_url and rrf_score.
        max_candidates: Maximum number of candidates to send to the LLM.

    Returns:
        Candidates list with adjusted rrf_score values.  On any failure,
        returns the original list unchanged.
    """
    if not detail_terms or not candidates:
        return candidates

    settings = get_settings()
    if not settings.vision_reranker_enabled:
        return candidates
    if not settings.google_api_key:
        logger.info("Vision reranker skipped: no GOOGLE_API_KEY")
        return candidates

    detail = ", ".join(detail_terms)
    top_n = min(max_candidates, len(candidates))
    top_candidates = candidates[:top_n]

    t_start = time.time()

    # Step 1: Download images in parallel
    t_dl_start = time.time()
    downloaded = _download_images_parallel(top_candidates)
    t_dl_end = time.time()

    if not downloaded:
        logger.warning("Vision reranker: no images downloaded successfully")
        return candidates

    # Step 2: Build prompt and call Gemini
    items = [(idx, cand) for idx, cand, _ in downloaded]
    images = [jpeg for _, _, jpeg in downloaded]
    prompt = _build_prompt(detail, items)

    t_llm_start = time.time()
    scores = _call_gemini(
        prompt=prompt,
        images=images,
        items=items,
        model=settings.vision_reranker_model,
        api_key=settings.google_api_key,
    )
    t_llm_end = time.time()

    if not scores:
        logger.warning("Vision reranker: no scores returned from Gemini")
        return candidates

    # Step 3: Apply score adjustments
    boosted = 0
    demoted = 0
    for idx, (llm_score, reason) in scores.items():
        if 0 <= idx < len(candidates):
            factor = _llm_score_to_factor(llm_score)
            old_rrf = candidates[idx].get("rrf_score", 0) or 0
            candidates[idx]["rrf_score"] = old_rrf * factor
            candidates[idx]["llm_detail_score"] = llm_score
            candidates[idx]["llm_detail_reason"] = reason
            if factor > 1.0:
                boosted += 1
            elif factor < 1.0:
                demoted += 1

    # Step 4: Re-sort by adjusted rrf_score
    candidates.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)

    t_end = time.time()

    logger.info(
        "Vision reranker completed",
        detail=detail,
        images_sent=len(downloaded),
        images_scored=len(scores),
        boosted=boosted,
        demoted=demoted,
        download_ms=round((t_dl_end - t_dl_start) * 1000),
        llm_ms=round((t_llm_end - t_llm_start) * 1000),
        total_ms=round((t_end - t_start) * 1000),
    )

    return candidates
