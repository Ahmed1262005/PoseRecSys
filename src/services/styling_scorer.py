"""
Styling Metadata Scorer for Complete the Fit v3.4
=================================================

Replaces the GPT-4.1 LLM judge with deterministic scoring based on
pre-computed ``styling_metadata`` from the Gemini Vision extraction
pipeline (extractor v1.0.0.2).

Each product's ``styling_metadata`` contains:
  - ``ideal_bottom_profile`` / ``ideal_top_profile`` / ``ideal_outerwear_profile``
  - ``pairing_rules`` (volume, formality, texture, length balance)
  - ``hard_avoids`` and ``soft_avoids`` (free-text styling negatives)
  - ``formality_level`` (1-10 numeric)
  - ``pairing_flexibility_score`` / ``layering_flexibility_score``

Integration:
  Called from outfit_engine._score_category() after TATTOO scoring.
  Returns a ``styling_adj`` added to each candidate's TATTOO score.
  Range: approximately -0.15 to +0.10.

Graceful degradation:
  If source lacks v1.0.0.2 attributes → styling_adj = 0 for all candidates.
  If candidate lacks v1.0.0.2 attributes → only profile-match scoring (no
  vibe/appearance coherence bonus).
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from services.outfit_engine import AestheticProfile

logger = logging.getLogger(__name__)


# =============================================================================
# 1. CONSTANTS
# =============================================================================

# --- Profile match bonuses ---
CATEGORY_MATCH_BONUS: float = 0.04
FIT_MATCH_BONUS: float = 0.02
MATERIAL_MATCH_BONUS: float = 0.02
COLOR_MATCH_BONUS: float = 0.01
RISE_MATCH_BONUS: float = 0.01
LENGTH_MATCH_BONUS: float = 0.01

# --- Avoid penalties ---
PROFILE_AVOID_PENALTY: float = -0.08
HARD_AVOID_PENALTY: float = -0.12
SOFT_AVOID_PENALTY: float = -0.05

# --- Pairing rules ---
PAIRING_RULES_BONUS: float = 0.02
PAIRING_RULES_PENALTY: float = -0.04

# --- Vibe/appearance coherence ---
VIBE_COHERENCE_BONUS: float = 0.02

# --- Formality mismatch ---
FORMALITY_MISMATCH_PENALTY: float = -0.04  # large gap (>3 levels)

# Max positive / negative caps
MAX_POSITIVE: float = 0.10
MAX_NEGATIVE: float = -0.15


# =============================================================================
# 2. CATEGORY MAPPING
# =============================================================================

# Map target broad_category to the profile key in styling_metadata
_PROFILE_KEY_MAP = {
    "bottoms": "ideal_bottom_profile",
    "tops": "ideal_top_profile",
    "outerwear": "ideal_outerwear_profile",
    "dresses": "ideal_outerwear_profile",  # dresses use outerwear profile (what goes over them)
}

# L2 normalisation: collapse common variants for fuzzy matching
_L2_NORMALIZE = re.compile(r"[\s_-]+")


def _norm(text: str) -> str:
    """Lowercase, collapse whitespace/underscores/hyphens."""
    if not text:
        return ""
    return _L2_NORMALIZE.sub("_", text.strip().lower())


def _norm_set(items: Any) -> Set[str]:
    """Convert a list/value to a set of normalised strings."""
    if not items:
        return set()
    if isinstance(items, str):
        return {_norm(items)}
    if isinstance(items, (list, tuple)):
        return {_norm(str(v)) for v in items if v}
    return set()


# =============================================================================
# 3. FUZZY MATCHING FOR FREE-TEXT AVOIDS
# =============================================================================

def _fuzzy_avoid_match(
    avoid_text: str,
    candidate_l2: str,
    candidate_name: str,
    candidate_fabric: str,
    candidate_fit: str,
) -> bool:
    """Check if a free-text avoid string matches a candidate.

    Avoids are written as natural language like:
      "oversized denim shirt", "low-rise skinny jeans",
      "anything overly formal or dressy", "satin evening trousers"

    We do keyword overlap matching against candidate attributes.
    """
    if not avoid_text:
        return False
    avoid_lower = avoid_text.lower()
    avoid_words = set(re.findall(r"[a-z]+", avoid_lower))

    # Skip vague avoids like "anything overly formal" — too broad for keyword match
    vague_prefixes = {"anything", "nothing", "never", "too"}
    if avoid_words & vague_prefixes and len(avoid_words) <= 6:
        return False

    # Build candidate descriptor words
    cand_words: Set[str] = set()
    for val in (candidate_l2, candidate_name, candidate_fabric, candidate_fit):
        if val:
            cand_words.update(re.findall(r"[a-z]+", val.lower()))

    # Require at least 2 keyword overlaps for a match (avoid false positives)
    overlap = avoid_words & cand_words
    # Remove very common words that don't add signal
    _STOP = {"the", "a", "an", "and", "or", "with", "for", "in", "of", "on", "very"}
    overlap -= _STOP
    return len(overlap) >= 2


# =============================================================================
# 4. VOLUME / FORMALITY BALANCE CHECKS
# =============================================================================

_FIT_VOLUME = {
    "fitted": 1, "slim": 2, "regular": 3, "relaxed": 4, "oversized": 5,
}

_FORMALITY_MAP = {
    "casual": 1, "smart_casual": 2, "smart casual": 2,
    "business_casual": 3, "business casual": 3, "business": 3,
    "semi_formal": 4, "semi-formal": 4, "formal": 5,
}


def _check_volume_balance(
    source_fit: Optional[str],
    candidate_fit: Optional[str],
    rules: Dict[str, str],
) -> float:
    """Check volume balance between source and candidate.

    Returns bonus (+PAIRING_RULES_BONUS) or penalty (PAIRING_RULES_PENALTY)
    or 0 if not enough info.
    """
    if not source_fit or not candidate_fit:
        return 0.0
    sv = _FIT_VOLUME.get(source_fit.lower().strip(), 3)
    cv = _FIT_VOLUME.get(candidate_fit.lower().strip(), 3)
    diff = abs(sv - cv)

    # Ideal: some contrast (diff 1-2). Same tightness or same bagginess is bad.
    # Very extreme contrast (diff 4+) can also be off.
    if 1 <= diff <= 2:
        return PAIRING_RULES_BONUS
    if diff == 0 and sv <= 2:
        # Both fitted/slim — can look too tight
        return PAIRING_RULES_PENALTY * 0.5
    if diff >= 4:
        # Very extreme (fitted + oversized)
        return PAIRING_RULES_PENALTY * 0.5
    return 0.0


def _check_formality_balance(
    source_formality_level: int,
    candidate_formality: Optional[str],
) -> float:
    """Check formality alignment. Large gaps penalised."""
    if not candidate_formality:
        return 0.0
    cl = _FORMALITY_MAP.get(candidate_formality.lower().strip(), 2)
    diff = abs(source_formality_level - cl)
    if diff <= 1:
        return PAIRING_RULES_BONUS * 0.5  # well-aligned
    if diff >= 3:
        return FORMALITY_MISMATCH_PENALTY
    return 0.0


# =============================================================================
# 5. MAIN SCORER
# =============================================================================

@dataclass
class StylingMatchResult:
    """Result of styling metadata scoring for one candidate."""
    styling_adj: float = 0.0
    breakdown: Dict[str, float] = field(default_factory=dict)
    matched_profile_fields: List[str] = field(default_factory=list)
    avoid_hits: List[str] = field(default_factory=list)
    has_source_metadata: bool = False


def compute_styling_adjustment(
    source: "AestheticProfile",
    candidate: "AestheticProfile",
    target_category: str,
    source_styling_metadata: Optional[Dict[str, Any]] = None,
    candidate_appearance_tags: Optional[List[str]] = None,
    candidate_vibe_tags: Optional[List[str]] = None,
    source_appearance_tags: Optional[List[str]] = None,
    source_vibe_tags: Optional[List[str]] = None,
) -> StylingMatchResult:
    """Compute styling adjustment for a single source→candidate pair.

    Args:
        source: source product profile
        candidate: candidate product profile
        target_category: broad category of the candidate ('bottoms', 'tops', 'outerwear')
        source_styling_metadata: the source's styling_metadata JSONB (or None)
        candidate_appearance_tags: candidate's appearance_top_tags (or None)
        candidate_vibe_tags: candidate's vibe_tags (or None)
        source_appearance_tags: source's appearance_top_tags (or None)
        source_vibe_tags: source's vibe_tags (or None)

    Returns:
        StylingMatchResult with styling_adj and debug breakdown.
    """
    result = StylingMatchResult()

    if not source_styling_metadata:
        return result  # No v1.0.0.2 data for source → 0 adjustment

    result.has_source_metadata = True
    adj = 0.0
    breakdown: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # A. Profile match scoring
    # ------------------------------------------------------------------
    profile_key = _PROFILE_KEY_MAP.get(target_category)
    profile = source_styling_metadata.get(profile_key) if profile_key else None

    if profile and isinstance(profile, dict):
        cand_l2 = _norm(candidate.gemini_category_l2 or "")

        # A1. Category match
        profile_cats = _norm_set(profile.get("categories"))
        if profile_cats and cand_l2:
            if cand_l2 in profile_cats:
                adj += CATEGORY_MATCH_BONUS
                breakdown["category_match"] = CATEGORY_MATCH_BONUS
                result.matched_profile_fields.append("category")
            else:
                # Partial match: check if candidate L2 contains any profile category word
                for pc in profile_cats:
                    if pc in cand_l2 or cand_l2 in pc:
                        adj += CATEGORY_MATCH_BONUS * 0.5
                        breakdown["category_partial"] = CATEGORY_MATCH_BONUS * 0.5
                        result.matched_profile_fields.append("category_partial")
                        break

        # A2. Fit match
        profile_fits = _norm_set(profile.get("fits"))
        cand_fit = _norm(candidate.fit_type or "")
        if profile_fits and cand_fit and cand_fit in profile_fits:
            adj += FIT_MATCH_BONUS
            breakdown["fit_match"] = FIT_MATCH_BONUS
            result.matched_profile_fields.append("fit")

        # A3. Rise match (bottoms only)
        profile_rises = _norm_set(profile.get("rises"))
        cand_rise = _norm(candidate.rise or "")
        if profile_rises and cand_rise and cand_rise in profile_rises:
            adj += RISE_MATCH_BONUS
            breakdown["rise_match"] = RISE_MATCH_BONUS
            result.matched_profile_fields.append("rise")

        # A4. Length match
        profile_lengths = _norm_set(profile.get("lengths"))
        cand_length = _norm(candidate.length or "")
        if profile_lengths and cand_length and cand_length in profile_lengths:
            adj += LENGTH_MATCH_BONUS
            breakdown["length_match"] = LENGTH_MATCH_BONUS
            result.matched_profile_fields.append("length")

        # A5. Material match
        profile_materials = _norm_set(profile.get("material_traits"))
        cand_fabric = _norm(candidate.apparent_fabric or "")
        cand_material_family = _norm(candidate.material_family or "")
        if profile_materials and (cand_fabric or cand_material_family):
            mat_match = False
            for pm in profile_materials:
                if (cand_fabric and (pm in cand_fabric or cand_fabric in pm)) or \
                   (cand_material_family and (pm in cand_material_family or cand_material_family in pm)):
                    mat_match = True
                    break
            if mat_match:
                adj += MATERIAL_MATCH_BONUS
                breakdown["material_match"] = MATERIAL_MATCH_BONUS
                result.matched_profile_fields.append("material")

        # A6. Color match
        profile_colors = _norm_set(profile.get("color_traits"))
        cand_color = _norm(candidate.color_family or "")
        if profile_colors and cand_color:
            for pc in profile_colors:
                if pc in cand_color or cand_color in pc:
                    adj += COLOR_MATCH_BONUS
                    breakdown["color_match"] = COLOR_MATCH_BONUS
                    result.matched_profile_fields.append("color")
                    break

        # A7. Profile avoid penalty
        profile_avoids = profile.get("avoid") or []
        if isinstance(profile_avoids, str):
            profile_avoids = [profile_avoids]
        for av in profile_avoids:
            if _fuzzy_avoid_match(
                str(av),
                candidate.gemini_category_l2 or "",
                candidate.name or "",
                candidate.apparent_fabric or "",
                candidate.fit_type or "",
            ):
                adj += PROFILE_AVOID_PENALTY
                breakdown["profile_avoid"] = PROFILE_AVOID_PENALTY
                result.avoid_hits.append(f"profile: {av}")
                break  # one hit is enough

    # ------------------------------------------------------------------
    # B. Hard & soft avoids from styling_metadata root
    # ------------------------------------------------------------------
    hard_avoids = source_styling_metadata.get("hard_avoids") or []
    if isinstance(hard_avoids, str):
        hard_avoids = [hard_avoids]
    for av in hard_avoids:
        if _fuzzy_avoid_match(
            str(av),
            candidate.gemini_category_l2 or "",
            candidate.name or "",
            candidate.apparent_fabric or "",
            candidate.fit_type or "",
        ):
            adj += HARD_AVOID_PENALTY
            breakdown["hard_avoid"] = HARD_AVOID_PENALTY
            result.avoid_hits.append(f"hard: {av}")
            break

    soft_avoids = source_styling_metadata.get("soft_avoids") or []
    if isinstance(soft_avoids, str):
        soft_avoids = [soft_avoids]
    for av in soft_avoids:
        if _fuzzy_avoid_match(
            str(av),
            candidate.gemini_category_l2 or "",
            candidate.name or "",
            candidate.apparent_fabric or "",
            candidate.fit_type or "",
        ):
            adj += SOFT_AVOID_PENALTY
            breakdown["soft_avoid"] = SOFT_AVOID_PENALTY
            result.avoid_hits.append(f"soft: {av}")
            break

    # ------------------------------------------------------------------
    # C. Pairing rules checks
    # ------------------------------------------------------------------
    rules = source_styling_metadata.get("pairing_rules") or {}

    # C1. Volume balance
    vol_adj = _check_volume_balance(source.fit_type, candidate.fit_type, rules)
    if vol_adj:
        adj += vol_adj
        breakdown["volume_balance"] = vol_adj

    # C2. Formality balance
    source_form_level = source_styling_metadata.get("formality_level")
    if source_form_level is None:
        source_form_level = source.formality_level
    form_adj = _check_formality_balance(int(source_form_level), candidate.formality)
    if form_adj:
        adj += form_adj
        breakdown["formality_balance"] = form_adj

    # ------------------------------------------------------------------
    # D. Vibe / appearance coherence (both need v1.0.0.2)
    # ------------------------------------------------------------------
    if source_appearance_tags and candidate_appearance_tags:
        s_tags = set(t.lower() for t in source_appearance_tags if t)
        c_tags = set(t.lower() for t in candidate_appearance_tags if t)
        overlap = len(s_tags & c_tags)
        if overlap >= 2:
            adj += VIBE_COHERENCE_BONUS
            breakdown["appearance_coherence"] = VIBE_COHERENCE_BONUS

    if source_vibe_tags and candidate_vibe_tags:
        # Vibe tags are free-form phrases — do word-level overlap
        s_words = set()
        for t in source_vibe_tags:
            s_words.update(re.findall(r"[a-z]+", t.lower()))
        c_words = set()
        for t in candidate_vibe_tags:
            c_words.update(re.findall(r"[a-z]+", t.lower()))
        _STOP = {"the", "a", "an", "and", "or", "of", "with", "for", "in"}
        s_words -= _STOP
        c_words -= _STOP
        if len(s_words & c_words) >= 2 and "appearance_coherence" not in breakdown:
            adj += VIBE_COHERENCE_BONUS * 0.5
            breakdown["vibe_coherence"] = VIBE_COHERENCE_BONUS * 0.5

    # ------------------------------------------------------------------
    # E. Clamp to range
    # ------------------------------------------------------------------
    adj = max(MAX_NEGATIVE, min(MAX_POSITIVE, adj))
    result.styling_adj = round(adj, 4)
    result.breakdown = {k: round(v, 4) for k, v in breakdown.items()}
    return result


# =============================================================================
# 6. BATCH SCORER (for use in _score_category)
# =============================================================================

def score_candidates_batch(
    source: "AestheticProfile",
    candidates: List["AestheticProfile"],
    target_category: str,
    source_styling_metadata: Optional[Dict[str, Any]] = None,
    source_appearance_tags: Optional[List[str]] = None,
    source_vibe_tags: Optional[List[str]] = None,
    candidate_extras: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, StylingMatchResult]:
    """Score all candidates against the source's styling metadata.

    Args:
        source: source product profile
        candidates: list of candidate profiles
        target_category: 'bottoms', 'tops', 'outerwear'
        source_styling_metadata: source's styling_metadata JSONB
        source_appearance_tags: source's appearance_top_tags
        source_vibe_tags: source's vibe_tags
        candidate_extras: dict of candidate_pid -> {appearance_top_tags, vibe_tags}

    Returns:
        Dict of candidate product_id -> StylingMatchResult
    """
    results: Dict[str, StylingMatchResult] = {}
    extras = candidate_extras or {}

    for cand in candidates:
        pid = cand.product_id or ""
        cand_extra = extras.get(pid, {})
        result = compute_styling_adjustment(
            source=source,
            candidate=cand,
            target_category=target_category,
            source_styling_metadata=source_styling_metadata,
            candidate_appearance_tags=cand_extra.get("appearance_top_tags"),
            candidate_vibe_tags=cand_extra.get("vibe_tags"),
            source_appearance_tags=source_appearance_tags,
            source_vibe_tags=source_vibe_tags,
        )
        results[pid] = result

    if source_styling_metadata:
        n_positive = sum(1 for r in results.values() if r.styling_adj > 0)
        n_negative = sum(1 for r in results.values() if r.styling_adj < 0)
        n_neutral = sum(1 for r in results.values() if r.styling_adj == 0)
        logger.info(
            "StylingScorer [%s]: %d candidates — %d boosted, %d penalised, %d neutral",
            target_category, len(candidates), n_positive, n_negative, n_neutral,
        )

    return results
