"""
Session-Aware Search Reranker.

Applies user-profile boosts and session-state adjustments
to merged search results. Runs after Algolia + FashionCLIP merge.

Scoring components (shared with the feed pipeline):
- ProfileScorer:  11-dimension attribute-driven scoring (brand clusters,
  style, formality, fit/sleeve/length/neckline/rise, type, pattern,
  occasion, color-avoid, price, coverage hard-kills, category boosts).
  Search uses LIGHT positive cap (+0.10) vs feed (+0.50).
  Negative cap is -1.0 to allow coverage hard-kills through.
- ContextScorer:  age-affinity + weather/season scoring.
  Search uses reduced weight (0.20) vs feed (1.0).
- Session scoring: EMA-powered brand/type/attr/intent boosts.
  Search uses +/-0.08 cap vs feed's +/-0.15.

Diversity enforcement:
- Near-duplicate removal (size variants, sister-brand mapping, same-image)
- Greedy constrained selection with:
  - Brand cap (max 4 per brand, no brand repeats in top 5 positions)
  - Category proportional caps (tops 35%, bottoms 25%, dresses 18%,
    outerwear 10%, other 12%) — same as feed reranker
  - Combo dedup: (brand_group, category, color, fit) uniqueness
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from recs.feed_reranker import (
    CATEGORY_GROUP_MAP,
    DEFAULT_CATEGORY_PROPORTIONS,
    compute_category_caps,
)

if TYPE_CHECKING:
    from scoring.context import UserContext


# Session-based scoring (EMA-powered, self-normalizing)
# With EMA, raw scores are already in [-1, 1] — no per-field caps needed.
# We keep a light overall cap to avoid session scoring from completely
# overwhelming the RRF base score.
SESSION_SKIP_PENALTY = -0.08
MAX_SESSION_BOOST = 0.08  # Search cap (lighter than feed's 0.15)

# Diversity defaults for search results
MAX_PER_BRAND = 4
STRICT_DIVERSITY_POSITIONS = 5  # no brand repeats in top N positions


# ============================================================================
# Helper: Extract attribute values from a search result dict
# ============================================================================

_PREFIX_TO_FIELDS = {
    "color": ["primary_color", "colors"],
    "fit": ["fit_type", "fit"],
    "sleeve": ["sleeve_type"],
    "length": ["length"],
    "neckline": ["neckline"],
    "pattern": ["pattern"],
    "occasion": ["occasions"],
    "style": ["style_tags"],
    "material": ["materials"],
    "formality": ["formality"],
    "category": ["article_type", "broad_category"],
    "brand": ["brand"],
}


def _get_item_attr_values(item: dict, prefix: str) -> Set[str]:
    """Extract attribute values from a search result item by prefix type.

    Maps scoring key prefixes (e.g., "color", "fit") to the corresponding
    fields in a search result dict (e.g., "primary_color", "colors").

    Returns a set of lowercased string values.
    """
    fields = _PREFIX_TO_FIELDS.get(prefix, [])
    values: Set[str] = set()
    for field in fields:
        val = item.get(field)
        if isinstance(val, str) and val:
            values.add(val.lower())
        elif isinstance(val, list):
            values.update(v.lower() for v in val if isinstance(v, str) and v)
    return values


def _resolve_category_group_dict(item: dict) -> str:
    """Map a search result dict's broad_category to a canonical group name."""
    raw = (
        item.get("broad_category")
        or item.get("article_type")
        or ""
    ).lower().strip()
    return CATEGORY_GROUP_MAP.get(raw, "_other")


def _get_combo_key(item: dict) -> str:
    """Build attribute combo key for dedup.

    Items with the same (brand_group, category, color, fit) are 'too similar'.
    """
    brand = (item.get("brand") or "").lower()
    canon_brand = SessionReranker._SISTER_BRANDS.get(brand, brand)
    parts = [
        canon_brand,
        (item.get("broad_category") or item.get("article_type") or "").lower(),
        (item.get("color_family") or item.get("primary_color") or "").lower(),
        (item.get("fit_type") or item.get("fit") or "").lower(),
    ]
    key = "|".join(parts)
    # Don't dedup if most fields are empty
    if key.replace("|", "").strip() == "":
        return ""
    return key


# ============================================================================
# Reranker
# ============================================================================

class SessionReranker:
    """
    Rerank search results using session state, shared ProfileScorer,
    ContextScorer, and diversity constraints.
    """

    def rerank(
        self,
        results: List[dict],
        user_profile: Optional[Dict[str, Any]] = None,
        seen_ids: Optional[Set[str]] = None,
        max_per_brand: int = MAX_PER_BRAND,
        user_context: Optional["UserContext"] = None,
        session_scores: Optional[Any] = None,
    ) -> List[dict]:
        """
        Rerank results with session awareness.

        Args:
            results: Merged search results (each has rrf_score).
            user_profile: User's onboarding profile (flat dict with
                OnboardingProfile field names, as returned by
                ``WomenSearchEngine.load_user_profile()``).
            seen_ids: Product IDs already shown in this session.
            max_per_brand: Max items per brand for diversity (default 4).
            user_context: Context for age-affinity + weather scoring.
            session_scores: Live SessionScores from the recommendation pipeline.

        Returns:
            Reranked and filtered results.
        """
        if not results:
            return results

        # Step 1: Remove already-seen items
        if seen_ids:
            results = [r for r in results if r.get("product_id") not in seen_ids]

        # Step 2: Remove near-duplicates (size variants, cross-brand dupes)
        results = self._deduplicate(results)

        # Step 3: Apply profile scoring via shared ProfileScorer
        if user_profile:
            results = self._apply_profile_scoring(results, user_profile)

        # Step 3.5: Apply live session scoring (brand/type/attr/intent boosts)
        if session_scores:
            results = self._apply_session_scoring(results, session_scores)

        # Step 3.75: Apply context-aware scoring (age affinity + weather/season)
        if user_context:
            results = self._apply_context_scoring(results, user_context)

        # Step 4: Greedy constrained diversity selection
        if max_per_brand > 0:
            results = self._apply_constrained_diversity(results, max_per_brand)

        return results

    # Size-variant prefixes/suffixes to strip when comparing names
    _SIZE_PATTERNS = re.compile(
        r'\b(petite|plus\s*size|tall|curve|curvy|regular|short|long)\b',
        re.IGNORECASE,
    )
    # Known sister-brand groups (same parent company, often share products)
    _SISTER_BRANDS: Dict[str, str] = {}
    for _canonical, _aliases in [
        ("boohoo", ["boohoo", "nasty gal", "prettylittlething", "plt",
                     "missguided", "karen millen", "coast", "oasis",
                     "debenhams", "dorothy perkins", "wallis", "burton"]),
    ]:
        for _alias in _aliases:
            _SISTER_BRANDS[_alias] = _canonical

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize product name for dedup comparison.

        Strips size-variant prefixes (Petite, Plus Size, Tall, Curve),
        collapses whitespace, and lowercases.
        """
        n = SessionReranker._SIZE_PATTERNS.sub("", name)
        n = re.sub(r"\s+", " ", n).strip().lower()
        return n

    def _deduplicate(self, results: List[dict]) -> List[dict]:
        """Remove near-duplicate products, keeping the highest-scored version.

        Catches:
        1. Size variants: "Front Twist Sheath Dress" vs
           "Petite Front Twist Sheath Dress" (same brand, same base name)
        2. Cross-brand dupes: Same product listed under sister brands
           (e.g. Boohoo and Nasty Gal share inventory)
        3. Same image URL across any brands
        """
        deduped: List[dict] = []
        seen_keys: Set[str] = set()
        seen_images: Set[str] = set()

        for item in results:
            name = item.get("name") or ""
            brand = (item.get("brand") or "").lower()
            image_url = item.get("image_url") or ""

            # Normalize name (strip size prefixes)
            norm_name = self._normalize_name(name)

            # Map brand to canonical for sister-brand detection
            canon_brand = self._SISTER_BRANDS.get(brand, brand)

            # Key 1: canonical brand + normalized name
            name_key = f"{canon_brand}|{norm_name}"

            # Key 2: image URL (catches cross-brand dupes with same photo)
            img_key = image_url.strip() if image_url else ""

            # Skip if we've seen this name-key or image before
            is_dupe = False
            if name_key in seen_keys:
                is_dupe = True
            if img_key and img_key in seen_images:
                is_dupe = True

            if not is_dupe:
                deduped.append(item)
                seen_keys.add(name_key)
                if img_key:
                    seen_images.add(img_key)

        return deduped

    def _apply_profile_scoring(
        self,
        results: List[dict],
        profile: Dict[str, Any],
    ) -> List[dict]:
        """Apply profile scoring via the shared ProfileScorer.

        Delegates to ``scoring.profile_scorer.ProfileScorer`` which provides
        11-dimension scoring: brand (with clusters + openness), style tags,
        formality, V3 category-aware fit/sleeve/length/neckline/rise, type
        preferences, pattern, occasion, color-avoid, price, coverage
        hard-kills, and category boosts.

        The profile dict should use flat OnboardingProfile field names
        (as returned by ``WomenSearchEngine.load_user_profile()``).
        """
        try:
            from scoring.profile_scorer import ProfileScorer, ProfileScoringConfig
            # Search uses light personalization: +0.10 positive cap
            # (feed uses +0.50/-2.0). max_negative must allow coverage
            # hard-kill (-1.0) to pass through — personality penalties
            # won't exceed -0.10 anyway since individual weights are small.
            search_config = ProfileScoringConfig(
                max_positive=0.10,
                max_negative=-1.0,
                coverage_kill_penalty=-1.0,
            )
            scorer = ProfileScorer(config=search_config)
            scorer.score_items(results, profile, score_field="rrf_score")
            # Re-sort by adjusted score
            results.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
        except Exception:
            # Non-fatal — if ProfileScorer import/execution fails, skip
            pass
        return results

    def _apply_session_scoring(
        self,
        results: List[dict],
        session_scores: Any,
    ) -> List[dict]:
        """Apply live session-learned preferences to search results.

        Uses EMA-based PreferenceState scores from the SessionScores object
        to boost/demote results based on real-time user behavior.

        With EMA, scores are self-normalizing (bounded to [-1, 1]),
        so we don't need per-field caps — just a light overall cap.

        Runs as Step 3.5 (after profile scoring, before context scoring).
        """
        try:
            skipped = getattr(session_scores, "skipped_ids", None) or set()
            # Use get_score() for EMA-aware reading (works with PreferenceState)
            _get_score = getattr(session_scores, "get_score", None)
        except Exception:
            return results  # Malformed session_scores, skip gracefully

        # If session_scores doesn't have get_score(), it's not a SessionScores
        if _get_score is None:
            return results

        for item in results:
            adjustment = 0.0
            product_id = item.get("product_id", "")
            brand = (item.get("brand") or "").lower()
            item_type = (
                item.get("article_type") or item.get("category") or ""
            ).lower()

            # Skip penalty
            if product_id in skipped:
                adjustment += SESSION_SKIP_PENALTY

            # Brand affinity from session (EMA score, already in [-1, 1])
            if brand:
                brand_score = _get_score("brand", brand)
                adjustment += brand_score * 0.10

            # Type affinity from session
            if item_type:
                type_score = _get_score("type", item_type)
                adjustment += type_score * 0.08

            # Attribute match from session
            # Check each prefix against item attribute values
            attr_boost = 0.0
            for prefix in ("color", "fit", "pattern", "occasion", "style",
                           "sleeve", "length", "neckline", "material",
                           "formality"):
                item_vals = _get_item_attr_values(item, prefix)
                for val in item_vals:
                    attr_key = f"{prefix}:{val}"
                    score = _get_score("attr", attr_key)
                    if score != 0.0:
                        attr_boost += score * 0.05
            adjustment += attr_boost

            # Search intent match
            intent_boost = 0.0
            for prefix in ("color", "fit", "pattern", "occasion", "style",
                           "sleeve", "length", "neckline", "material",
                           "formality", "type", "category"):
                item_vals = _get_item_attr_values(item, prefix)
                for val in item_vals:
                    intent_key = f"{prefix}:{val}"
                    score = _get_score("intent", intent_key)
                    if score != 0.0:
                        intent_boost += score * 0.06
            adjustment += intent_boost

            # Light overall cap (EMA is self-normalizing but cumulative
            # across many matching attributes could exceed)
            adjustment = max(-MAX_SESSION_BOOST, min(MAX_SESSION_BOOST, adjustment))

            item["rrf_score"] = item.get("rrf_score", 0) + adjustment
            item["session_adjustment"] = adjustment

        # Re-sort by adjusted score
        results.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
        return results

    def _apply_context_scoring(
        self,
        results: List[dict],
        user_context: "UserContext",
    ) -> List[dict]:
        """Apply context-aware scoring (age affinity + weather/season).

        Uses the shared ``ContextScorer`` from ``src/scoring/``.
        Adds adjustment to ``rrf_score`` and re-sorts.
        """
        try:
            from scoring import ContextScorer
            scorer = ContextScorer()
            # Search uses reduced weight (0.20) vs feed (1.0)
            scorer.score_items(
                results, user_context,
                score_field="rrf_score",
                weight=0.20,
            )
            # Re-sort by adjusted rrf_score
            results.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
        except Exception:
            # Non-fatal — if scoring module fails, just skip
            pass
        return results

    def _apply_constrained_diversity(
        self,
        results: List[dict],
        max_per_brand: int,
    ) -> List[dict]:
        """Greedy constrained selection with brand, category, and combo diversity.

        Mirrors the feed reranker's approach but adapted for search result dicts:
        1. Brand cap — max N items per brand
        2. Position-aware brand diversity — no brand repeats in top positions
        3. Category proportional caps — tops/bottoms/dresses/outerwear allocation
        4. Combo dedup — skip items too similar in (brand_group, category, color, fit)

        Falls back to a simple brand-cap filter if no category data is available.
        """
        if not results:
            return results

        target_size = len(results)
        category_caps = compute_category_caps(target_size, DEFAULT_CATEGORY_PROPORTIONS)

        brand_counts: Dict[str, int] = defaultdict(int)
        group_counts: Dict[str, int] = defaultdict(int)
        used_combos: Set[str] = set()
        diverse: List[dict] = []

        for item in results:
            brand = (item.get("brand") or "unknown").lower()
            group = _resolve_category_group_dict(item)
            position = len(diverse)

            # Brand cap
            if brand_counts[brand] >= max_per_brand:
                continue

            # Strict diversity in top positions: no brand repeats
            if position < STRICT_DIVERSITY_POSITIONS and brand_counts[brand] > 0:
                continue

            # Category proportional cap
            if group == "_other":
                cap = max(category_caps.get(group, 6), 6)
            else:
                cap = category_caps.get(group, 6)
            if group_counts[group] >= cap:
                continue

            # Combo dedup — skip items too similar to already-selected ones
            combo = _get_combo_key(item)
            if combo and combo in used_combos:
                continue

            # Item passes all constraints
            diverse.append(item)
            brand_counts[brand] += 1
            group_counts[group] += 1
            if combo:
                used_combos.add(combo)

        # If strict constraints filtered too aggressively, backfill from
        # remaining items — relax category caps and combo dedup but still
        # respect brand caps (the hard diversity constraint).
        if len(diverse) < min(target_size, 20):
            used_ids = {r.get("product_id") for r in diverse}
            for item in results:
                if len(diverse) >= target_size:
                    break
                pid = item.get("product_id")
                if pid in used_ids:
                    continue
                brand = (item.get("brand") or "unknown").lower()
                if brand_counts[brand] >= max_per_brand:
                    continue  # Never exceed brand cap, even in backfill
                diverse.append(item)
                used_ids.add(pid)
                brand_counts[brand] += 1

        return diverse
