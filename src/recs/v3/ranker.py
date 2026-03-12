"""
V3 Feed Ranker — single-pass scorer for all candidates.

Computes final_score = weighted sum of normalized components * penalties.

9 real signal components (SASRec dropped):
  retrieval, preference, session, embedding, complement,
  context, novelty, trending, exploration

Two weight profiles: warm (session-rich) and cold (new user).
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from recs.brand_clusters import get_cluster_for_item, DEFAULT_CLUSTER
from recs.models import Candidate
from recs.v3.models import CandidatePool, ScoringMeta, SessionProfile

logger = logging.getLogger(__name__)


# =========================================================================
# Weight profiles
# =========================================================================

@dataclass
class WeightProfile:
    """Scoring weights for a single component profile."""
    retrieval: float
    preference: float
    session: float
    embedding: float
    complement: float
    context: float
    novelty: float
    trending: float
    exploration: float

    def total(self) -> float:
        return (
            self.retrieval + self.preference + self.session +
            self.embedding + self.complement + self.context +
            self.novelty + self.trending + self.exploration
        )


# Warm weights (session-rich user): sum = 1.0
WARM_WEIGHTS = WeightProfile(
    retrieval=0.15, preference=0.25, session=0.25,
    embedding=0.10, complement=0.05, context=0.08,
    novelty=0.05, trending=0.03, exploration=0.04,
)

# Cold weights (new user): sum = 1.0
COLD_WEIGHTS = WeightProfile(
    retrieval=0.25, preference=0.35, session=0.05,
    embedding=0.10, complement=0.0, context=0.07,
    novelty=0.05, trending=0.04, exploration=0.10,
)


# =========================================================================
# Penalty config
# =========================================================================

@dataclass
class PenaltyConfig:
    """Multiplicative penalty factors."""
    repeated_brand: float = 0.85
    repeated_cluster: float = 0.92
    repeated_category: float = 0.80
    fatigue: float = 0.95
    price_mismatch: float = 0.90


DEFAULT_PENALTIES = PenaltyConfig()


# =========================================================================
# Ranker
# =========================================================================

class FeedRanker:
    """
    Single-pass scorer for all candidates.

    Computes final_score = weighted sum of normalized components * penalties.

    External scorers (profile, session, context) are injected at init
    so the ranker itself is stateless and testable.
    """

    def __init__(
        self,
        profile_scorer: Any = None,
        scoring_engine: Any = None,
        context_scorer: Any = None,
        penalties: PenaltyConfig = None,
    ) -> None:
        self.profile_scorer = profile_scorer
        self.scoring_engine = scoring_engine
        self.context_scorer = context_scorer
        self.penalties = penalties or DEFAULT_PENALTIES

    def rank(
        self,
        candidates: List[Candidate],
        user_profile: Any = None,
        session: Optional[SessionProfile] = None,
        session_scores: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        is_warm: bool = False,
        eligibility_penalties: Optional[Dict[str, float]] = None,
        soft_preferences: Optional[Dict[str, Any]] = None,
    ) -> List[Candidate]:
        """Score and sort candidates descending by final_score."""
        weights = WARM_WEIGHTS if is_warm else COLD_WEIGHTS
        elig_penalties = eligibility_penalties or {}

        # Compute blended session scores if scoring engine available
        blended = None
        if self.scoring_engine and session_scores:
            try:
                blended = self.scoring_engine.blend_scores(session_scores)
            except Exception:
                blended = None

        for candidate in candidates:
            # Component scores
            components = self._compute_component_scores(
                candidate, weights, user_profile, session,
                session_scores, blended, context,
            )

            # Soft preference affinity boost: items matching active filters
            # (e.g. include_style_tags=boho) get a scoring uplift on the
            # preference component.  This makes the filter a ranking signal,
            # not just a binary pass/fail gate.
            if soft_preferences:
                affinity = self._compute_preference_affinity(candidate, soft_preferences)
                if affinity > 0:
                    # Blend: 70% original preference + 30% affinity match
                    orig = components.get("preference", 0.0)
                    components["preference"] = _clamp01(0.7 * orig + 0.3 * affinity)

            # Weighted sum
            score = 0.0
            for comp_name, comp_score in components.items():
                w = getattr(weights, comp_name, 0.0)
                score += w * comp_score

            # Apply penalties
            penalty_mult = 1.0

            # Brand fatigue
            if session and candidate.brand:
                brand_count = session.brand_exposure.get(candidate.brand, 0)
                if brand_count > 0:
                    penalty_mult *= self.penalties.repeated_brand ** min(brand_count, 5)

            # Cluster fatigue
            cluster_id = get_cluster_for_item(candidate.brand or "")
            if session and cluster_id != DEFAULT_CLUSTER:
                cluster_count = session.cluster_exposure.get(cluster_id, 0)
                if cluster_count > 0:
                    penalty_mult *= self.penalties.repeated_cluster ** min(cluster_count, 5)

            # Category fatigue
            if session and candidate.article_type:
                cat_count = session.category_exposure.get(candidate.article_type, 0)
                if cat_count > 0:
                    penalty_mult *= self.penalties.repeated_category

            # Price mismatch
            if user_profile and candidate.price:
                if not self._price_in_comfort(candidate.price, user_profile):
                    penalty_mult *= self.penalties.price_mismatch

            # Eligibility penalty (from eligibility filter)
            elig_pen = elig_penalties.get(candidate.item_id, 0.0)
            if elig_pen > 0:
                penalty_mult *= (1.0 - elig_pen)

            base = score * penalty_mult

            # Search affinity: additive boost when candidate matches
            # recently searched categories / article_types / brands.
            search_boost = 0.0
            if session and session.search_intents:
                search_boost = self._compute_search_affinity(
                    candidate, session,
                )

            candidate.final_score = _clamp01(base + search_boost)

        # Sort descending
        candidates.sort(key=lambda x: getattr(x, "final_score", 0.0), reverse=True)
        return candidates

    def _compute_component_scores(
        self,
        candidate: Candidate,
        weights: WeightProfile,
        user_profile: Any,
        session: Optional[SessionProfile],
        session_scores: Optional[Any],
        blended: Optional[Any],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Compute normalized [0, 1] scores for each component.

        Missing scorers produce 0.0 for their component.
        """
        scores: Dict[str, float] = {}

        # Retrieval score (from source)
        scores["retrieval"] = _clamp01(
            getattr(candidate, "retrieval_score", 0.0) or 0.0
        )

        # Preference score (from profile scorer)
        if self.profile_scorer and user_profile:
            try:
                scores["preference"] = _clamp01(
                    self.profile_scorer.score_item(candidate, user_profile)
                )
            except Exception:
                scores["preference"] = 0.0
        else:
            scores["preference"] = 0.0

        # Session score
        if blended and candidate.item_id in blended:
            scores["session"] = _clamp01(blended[candidate.item_id])
        elif session_scores:
            try:
                # Try direct scoring
                scores["session"] = _clamp01(
                    getattr(session_scores, "score", lambda _: 0.0)(candidate)
                )
            except Exception:
                scores["session"] = 0.0
        else:
            scores["session"] = 0.0

        # Embedding score
        scores["embedding"] = _clamp01(
            getattr(candidate, "embedding_score", 0.0) or 0.0
        )

        # Complement score (warm only)
        scores["complement"] = 0.0

        # Context score
        if self.context_scorer and context:
            try:
                scores["context"] = _clamp01(
                    self.context_scorer.score_item(candidate, context)
                )
            except Exception:
                scores["context"] = 0.0
        else:
            scores["context"] = 0.0

        # Novelty score
        is_new = getattr(candidate, "is_new", False)
        is_on_sale = getattr(candidate, "is_on_sale", False)
        if is_new and is_on_sale:
            scores["novelty"] = 1.0
        elif is_new:
            scores["novelty"] = 0.8
        elif is_on_sale:
            scores["novelty"] = 0.6
        else:
            scores["novelty"] = 0.3

        # Trending score
        scores["trending"] = 0.0

        # Exploration score (random jitter for diversity)
        scores["exploration"] = random.random() * 0.5 + 0.4

        return scores

    @staticmethod
    def _price_in_comfort(price: float, profile: Any) -> bool:
        """Check if price is within user's comfort band."""
        min_p = getattr(profile, "min_price", None)
        max_p = getattr(profile, "max_price", None)
        if min_p is not None and price < min_p:
            return False
        if max_p is not None and price > max_p:
            return False
        return True

    @staticmethod
    def _compute_preference_affinity(
        candidate: Candidate,
        soft_prefs: Dict[str, Any],
    ) -> float:
        """Compute 0.0–1.0 affinity score for how well a candidate matches
        the user's active soft preference filters.

        Each matched include_* dimension adds to the score; each matched
        exclude_* dimension subtracts.  The result is the fraction of
        dimensions that matched positively.

        Mapping of soft_pref keys → candidate attributes:
          include_style_tags / exclude_style_tags   → candidate.style_tags (list)
          include_formality  / exclude_formality     → candidate.formality (str)
          include_color_family / exclude_color_family → candidate.color_family (str)
          include_silhouette / exclude_silhouette     → candidate.silhouette (str)
          include_coverage   / exclude_coverage       → candidate.coverage_level (str)
          include_fit        / exclude_fit            → candidate.fit (str)
          include_sleeves    / exclude_sleeves        → candidate.sleeve_length (str)
          include_seasons    / exclude_seasons        → candidate.seasons (list)
          include_length     / exclude_length         → candidate.length (str)
          include_neckline   / exclude_neckline       → candidate.neckline (str)
          include_patterns   / exclude_patterns       → candidate.pattern (str)
          include_materials  / exclude_materials      → candidate.material (str)
        """
        # Attribute map: (include_key, exclude_key, attr_name, is_list)
        _DIMS = [
            ("include_style_tags",   "exclude_style_tags",   "style_tags",      True),
            ("include_formality",    "exclude_formality",    "formality",        False),
            ("include_color_family", "exclude_color_family", "color_family",     False),
            ("include_silhouette",   "exclude_silhouette",   "silhouette",       False),
            ("include_coverage",     "exclude_coverage",     "coverage_level",   False),
            ("include_fit",          "exclude_fit",          "fit",              False),
            ("include_sleeves",      "exclude_sleeves",      "sleeve_length",    False),
            ("include_seasons",      "exclude_seasons",      "seasons",          True),
            ("include_length",       "exclude_length",       "length",           False),
            ("include_neckline",     "exclude_neckline",     "neckline",         False),
            ("include_patterns",     "exclude_patterns",     "pattern",          False),
            ("include_materials",    "exclude_materials",    "material",         False),
        ]

        hits = 0
        checks = 0

        for inc_key, exc_key, attr, is_list in _DIMS:
            inc_vals = soft_prefs.get(inc_key)
            exc_vals = soft_prefs.get(exc_key)
            if not inc_vals and not exc_vals:
                continue

            # Resolve candidate value(s)
            raw = getattr(candidate, attr, None)
            if is_list:
                item_vals = {v.lower() for v in (raw or [])}
            else:
                item_vals = {raw.lower()} if raw else set()

            if not item_vals:
                continue  # no data on candidate — skip dimension

            checks += 1

            # Include match: any overlap = hit
            if inc_vals:
                inc_set = {v.lower() for v in inc_vals}
                if item_vals & inc_set:
                    hits += 1
                    continue  # counted, don't double-count exclude

            # Exclude match: overlap = anti-hit (0 for this dimension)
            if exc_vals:
                exc_set = {v.lower() for v in exc_vals}
                if item_vals & exc_set:
                    # Matched an exclude — no hit
                    continue
                else:
                    # Doesn't match any exclude — mild positive
                    hits += 1

        if checks == 0:
            return 0.0
        return hits / checks

    @staticmethod
    def _compute_search_affinity(
        candidate: Candidate,
        session: SessionProfile,
    ) -> float:
        """Additive score boost when a candidate matches recent searches.

        For each recent search intent (most-recent first, recency decay
        0.7^i), checks three dimensions:

            category match   → 0.06
            article_type match → 0.12
            brand match      → 0.04

        Returns ``min(best_boost, 0.18)`` so a single perfect search
        match gives up to +0.18 on final_score.
        """
        DECAY = 0.7
        CAT_W = 0.06
        TYPE_W = 0.12
        BRAND_W = 0.04
        CAP = 0.18

        cand_cat = (getattr(candidate, "category", None) or "").lower()
        cand_type = (getattr(candidate, "article_type", None) or "").lower()
        cand_brand = (getattr(candidate, "brand", None) or "").lower()

        best = 0.0

        for i, intent in enumerate(reversed(session.search_intents)):
            recency = DECAY ** i
            boost = 0.0

            if cand_cat and cand_cat in intent.get("cats", []):
                boost += CAT_W
            if cand_type and cand_type in intent.get("types", []):
                boost += TYPE_W
            if cand_brand and cand_brand in intent.get("brands", []):
                boost += BRAND_W

            best = max(best, recency * boost)

        return min(best, CAP)

    def rerank_pool_from_meta(
        self,
        pool: CandidatePool,
        session: SessionProfile,
        session_scores: Optional[Any] = None,
    ) -> None:
        """
        Re-score remaining pool items using ScoringMeta + session fatigue.

        Called when should_rerank() is true. Updates scores and
        reorders remaining items. No database queries.

        Drift signals used (all from ScoringMeta + SessionProfile):
          - retrieval_score: base relevance from source RPC
          - brand fatigue: repeated_brand penalty from session.brand_exposure
          - cluster fatigue: repeated_cluster penalty from session.cluster_exposure
          - category fatigue: repeated_category penalty from session.category_exposure
          - session_scores: optional blended scores from scoring_engine
        """
        # Only re-rank remaining (unserved) items
        start = pool.served_count
        remaining_ids = pool.ordered_ids[start:]

        if not remaining_ids:
            return

        # Compute blended session scores (from external scoring engine, if any)
        blended = None
        if self.scoring_engine and session_scores:
            try:
                blended = self.scoring_engine.blend_scores(session_scores)
            except Exception:
                blended = None

        new_scores = {}
        for item_id in remaining_ids:
            meta = pool.meta.get(item_id)
            if meta is None:
                new_scores[item_id] = 0.0
                continue

            # Base score from retrieval
            retrieval = meta.retrieval_score

            # Session-aligned score from blended signals (0.0 if no scoring engine)
            session_score = self._score_from_meta(
                blended, meta.brand, meta.cluster_id,
                meta.broad_category, meta.article_type,
            )

            score = 0.6 * retrieval + 0.4 * session_score

            # Apply session fatigue penalties (mirrors rank() penalties)
            penalty_mult = 1.0

            # Brand fatigue
            brand_lower = meta.brand.lower() if meta.brand else ""
            if brand_lower:
                brand_count = session.brand_exposure.get(brand_lower, 0)
                if brand_count > 0:
                    penalty_mult *= self.penalties.repeated_brand ** min(brand_count, 5)

            # Cluster fatigue
            cluster_id = meta.cluster_id or ""
            if cluster_id and cluster_id != DEFAULT_CLUSTER:
                cluster_count = session.cluster_exposure.get(cluster_id, 0)
                if cluster_count > 0:
                    penalty_mult *= self.penalties.repeated_cluster ** min(cluster_count, 5)

            # Category fatigue
            article_type = meta.article_type or ""
            if article_type:
                cat_count = session.category_exposure.get(article_type, 0)
                if cat_count > 0:
                    penalty_mult *= self.penalties.repeated_category

            base = score * penalty_mult

            # Search affinity boost from ScoringMeta fields
            search_boost = self._search_affinity_from_meta(
                session, meta.broad_category, meta.article_type, meta.brand,
            )

            new_scores[item_id] = max(0.0, base + search_boost)

        # Re-sort remaining IDs by new scores
        remaining_ids.sort(
            key=lambda x: new_scores.get(x, 0.0), reverse=True
        )

        # Update pool
        pool.ordered_ids = pool.ordered_ids[:start] + remaining_ids
        pool.scores.update(new_scores)
        pool.last_rerank_action_seq = session.action_seq

    def _score_from_meta(
        self,
        blended: Optional[Any],
        brand: str,
        cluster_id: str,
        broad_category: str,
        article_type: str,
    ) -> float:
        """
        Compute session-aligned score from ScoringMeta fields.

        Returns same-direction score as full scoring (higher = better match).
        """
        score = 0.0

        if blended:
            # Check cluster match
            cluster_key = f"cluster:{cluster_id}" if cluster_id else None
            if cluster_key and cluster_key in blended:
                score += 0.3 * blended[cluster_key]

            # Check brand match
            brand_key = f"brand:{brand}" if brand else None
            if brand_key and brand_key in blended:
                score += 0.3 * blended[brand_key]

            # Check type match
            type_key = f"type:{article_type}" if article_type else None
            if type_key and type_key in blended:
                score += 0.4 * blended[type_key]

        return _clamp01(score)

    def score_item_from_meta(
        self,
        session_scores: Optional[Any],
        brand: str,
        cluster_id: str,
        broad_category: str,
        article_type: str,
    ) -> float:
        """
        Public API for pool re-ranking: score a single item from meta only.

        Convenience wrapper around _score_from_meta.
        """
        blended = None
        if self.scoring_engine and session_scores:
            try:
                blended = self.scoring_engine.blend_scores(session_scores)
            except Exception:
                blended = None
        return self._score_from_meta(blended, brand, cluster_id, broad_category, article_type)

    @staticmethod
    def _search_affinity_from_meta(
        session: SessionProfile,
        broad_category: str,
        article_type: str,
        brand: str,
    ) -> float:
        """Same logic as _compute_search_affinity but using ScoringMeta fields.

        Used on the rerank-from-meta path where full Candidate objects
        are not available.
        """
        if not session.search_intents:
            return 0.0

        DECAY = 0.7
        CAT_W = 0.06
        TYPE_W = 0.12
        BRAND_W = 0.04
        CAP = 0.18

        cat_l = (broad_category or "").lower()
        type_l = (article_type or "").lower()
        brand_l = (brand or "").lower()

        best = 0.0
        for i, intent in enumerate(reversed(session.search_intents)):
            recency = DECAY ** i
            boost = 0.0
            if cat_l and cat_l in intent.get("cats", []):
                boost += CAT_W
            if type_l and type_l in intent.get("types", []):
                boost += TYPE_W
            if brand_l and brand_l in intent.get("brands", []):
                boost += BRAND_W
            best = max(best, recency * boost)

        return min(best, CAP)


def _clamp01(v: float) -> float:
    """Clamp value to [0.0, 1.0]."""
    return max(0.0, min(1.0, v))
