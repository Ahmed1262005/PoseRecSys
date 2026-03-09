"""
V3 Eligibility Filter — consolidated 10+1 check filter.

Ported from V2's feasibility_filter.py with additions:
  - Check 7b: include_brands whitelist (bidirectional substring match)
  - Check 8b: include_rise / exclude_rise
  - Penalty-based soft filtering for unknown article types

Imports shared constants from recs.feasibility_filter.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from recs.feasibility_filter import (
    ARTICLE_TYPE_CANON,
    ATHLETIC_BRANDS,
    OCCASIONS_BLOCKING_ATHLETIC_BRANDS,
    OCCASION_ALLOWED,
    UNIVERSAL_BLOCKS,
    USER_EXCLUSION_MAPPING,
    canonicalize_article_type,
    canonicalize_name,
    get_broad_category,
)
from recs.models import Candidate

logger = logging.getLogger(__name__)


@dataclass
class EligibilityResult:
    """Result of a single-item eligibility check."""
    passes: bool
    reason: Optional[str] = None
    penalty: float = 0.0
    failed_rules: List[str] = field(default_factory=list)


class EligibilityFilter:
    """
    Consolidated eligibility filter for V3 feed.

    10 core checks + Check 7b (include_brands whitelist):
      1. Already shown (shown_set)
      2. Hidden by user (hidden_ids)
      3. Explicit negative brand (negative_brands)
      4a. User exclusions (no_sleeveless, no_crop, no_tanks, no_revealing, no_athletic)
      4b. Explicit article type exclusion
      5. Universal blocks (occasion-based)
      6. Occasion allowed list
      7a. Brand exclusion (exclude_brands)
      7b. Brand inclusion whitelist (include_brands)
      8a. Color exclusion
      8b. Rise inclusion / exclusion
      9. Unknown article type penalty (soft)
    """

    def __init__(self) -> None:
        pass

    def check(
        self,
        candidate: Candidate,
        occasions: Optional[List[str]] = None,
        user_exclusions: Optional[List[str]] = None,
        excluded_article_types: Optional[List[str]] = None,
        exclude_colors: Optional[List[str]] = None,
        exclude_brands: Optional[List[str]] = None,
        include_brands: Optional[List[str]] = None,
        include_rise: Optional[List[str]] = None,
        exclude_rise: Optional[List[str]] = None,
        hidden_ids: Optional[Set[str]] = None,
        negative_brands: Optional[Set[str]] = None,
        shown_set: Optional[Set[str]] = None,
    ) -> EligibilityResult:
        """
        Check if a candidate passes all eligibility constraints.

        Returns EligibilityResult with pass/fail, reason, and penalty.
        """
        item_id = candidate.item_id
        brand = (candidate.brand or "").lower()
        canonical_type = canonicalize_article_type(
            getattr(candidate, "article_type", None) or "unknown"
        )
        name_lower = canonicalize_name(candidate.name) if candidate.name else ""
        failed_rules: List[str] = []
        penalty = 0.0

        # Check 1: Already shown
        if shown_set and item_id in shown_set:
            return EligibilityResult(
                passes=False, reason="already_shown",
                failed_rules=["shown_set"],
            )

        # Check 2: Hidden by user
        if hidden_ids and item_id in hidden_ids:
            return EligibilityResult(
                passes=False, reason="hidden_by_user",
                failed_rules=["hidden_ids"],
            )

        # Check 3: Explicit negative brand
        if negative_brands and brand:
            for nb in negative_brands:
                nb_lower = nb.lower()
                if nb_lower in brand or brand in nb_lower:
                    return EligibilityResult(
                        passes=False,
                        reason=f"negative_brand:{nb}",
                        failed_rules=["explicit_negative_brand"],
                    )

        # Check 4a: User exclusion preferences
        if user_exclusions:
            ux_set = set(user_exclusions)

            # no_sleeveless
            if "no_sleeveless" in ux_set:
                sleeveless_types = USER_EXCLUSION_MAPPING.get("no_sleeveless", [])
                if canonical_type in sleeveless_types:
                    return EligibilityResult(
                        passes=False,
                        reason=f"sleeveless_type:{canonical_type}",
                        failed_rules=["no_sleeveless"],
                    )
                sleeve = (getattr(candidate, "sleeve", None) or "").lower()
                if sleeve and "sleeveless" in sleeve:
                    return EligibilityResult(
                        passes=False,
                        reason=f"sleeveless_sleeve:{sleeve}",
                        failed_rules=["no_sleeveless"],
                    )

            # no_crop
            if "no_crop" in ux_set:
                crop_types = USER_EXCLUSION_MAPPING.get("no_crop", [])
                if canonical_type in crop_types:
                    return EligibilityResult(
                        passes=False,
                        reason="crop_top",
                        failed_rules=["no_crop"],
                    )

            # no_tanks
            if "no_tanks" in ux_set:
                tank_types = USER_EXCLUSION_MAPPING.get("no_tanks", [])
                if canonical_type in tank_types:
                    return EligibilityResult(
                        passes=False,
                        reason="tank_top",
                        failed_rules=["no_tanks"],
                    )

            # no_revealing
            if "no_revealing" in ux_set:
                skin_exp = getattr(candidate, "skin_exposure", None) or ""
                coverage = getattr(candidate, "coverage_level", None) or ""
                if skin_exp.lower() in ("high",) or coverage.lower() in ("minimal",):
                    return EligibilityResult(
                        passes=False,
                        reason=f"revealing:skin={skin_exp},coverage={coverage}",
                        failed_rules=["no_revealing"],
                    )
                revealing_types = USER_EXCLUSION_MAPPING.get("no_revealing", [])
                if canonical_type in revealing_types:
                    return EligibilityResult(
                        passes=False,
                        reason=f"revealing_type:{canonical_type}",
                        failed_rules=["no_revealing"],
                    )

            # no_athletic
            if "no_athletic" in ux_set:
                athletic_types = USER_EXCLUSION_MAPPING.get("no_athletic", [])
                if canonical_type in athletic_types:
                    return EligibilityResult(
                        passes=False,
                        reason=f"athletic_type:{canonical_type}",
                        failed_rules=["no_athletic"],
                    )

            # Generic user exclusion check
            for ux in ux_set:
                if ux.startswith("no_"):
                    continue  # Already handled above
                mapped = USER_EXCLUSION_MAPPING.get(ux, [])
                if canonical_type in mapped:
                    return EligibilityResult(
                        passes=False,
                        reason=f"user_exclusion:{ux}:{canonical_type}",
                        failed_rules=[ux],
                    )

        # Check 4b: Explicit article type exclusion
        if excluded_article_types:
            eat_lower = {e.lower().replace("-", "_").replace(" ", "_")
                         for e in excluded_article_types}
            if canonical_type in eat_lower:
                return EligibilityResult(
                    passes=False,
                    reason=f"excluded_article_type:{canonical_type}",
                    failed_rules=["explicit_article_type_exclusion"],
                )

        # Check 5: Universal blocks (occasion-based)
        if occasions:
            for occ in occasions:
                occ_lower = occ.lower()
                blocks = UNIVERSAL_BLOCKS.get(occ_lower, [])
                if canonical_type in blocks:
                    return EligibilityResult(
                        passes=False,
                        reason=f"universal_block:{occ_lower}:{canonical_type}",
                        failed_rules=["occasion_block"],
                    )

                # Check 6: Occasion allowed list
                allowed = OCCASION_ALLOWED.get(occ_lower)
                if allowed is not None and canonical_type not in allowed:
                    # Check by name too
                    if not any(a in name_lower for a in allowed if a):
                        return EligibilityResult(
                            passes=False,
                            reason=f"not_allowed:{occ_lower}:{canonical_type}",
                            failed_rules=["occasion_allowed_list"],
                        )

                # Check name-based blocking
                blocks_names = UNIVERSAL_BLOCKS.get(occ_lower, [])
                if any(b in name_lower for b in blocks_names if b):
                    return EligibilityResult(
                        passes=False,
                        reason=f"name_blocked:{occ_lower}",
                        failed_rules=["occasion_name_block"],
                    )

                # Check: Athletic brands blocked for certain occasions
                if occ_lower in OCCASIONS_BLOCKING_ATHLETIC_BRANDS:
                    if brand and any(
                        ab.lower() in brand or brand in ab.lower()
                        for ab in ATHLETIC_BRANDS
                    ):
                        return EligibilityResult(
                            passes=False,
                            reason=f"athletic_brand:{brand}",
                            failed_rules=["athletic_brand_block"],
                        )

        # Check 7a: Brand exclusion
        if exclude_brands and brand:
            for eb in exclude_brands:
                eb_lower = eb.lower()
                if eb_lower in brand or brand in eb_lower:
                    return EligibilityResult(
                        passes=False,
                        reason=f"excluded_brand:{eb}",
                        failed_rules=["brand_exclusion"],
                    )

        # Check 7b: Brand inclusion whitelist
        if include_brands:
            if not brand:
                return EligibilityResult(
                    passes=False,
                    reason="include_brands:no_brand_data",
                    failed_rules=["brand_inclusion"],
                )
            matched = False
            for ib in include_brands:
                ib_lower = ib.lower()
                if ib_lower in brand or brand in ib_lower:
                    matched = True
                    break
            if not matched:
                return EligibilityResult(
                    passes=False,
                    reason=f"not_in_include_brands:{brand}",
                    failed_rules=["brand_inclusion"],
                )

        # Check 8a: Color exclusion
        if exclude_colors:
            item_colors = [c.lower() for c in (candidate.colors or [])]
            color_family = (getattr(candidate, "color_family", None) or "").lower()

            for ec in exclude_colors:
                ec_lower = ec.lower()
                # Check colors array
                if any(ec_lower in ic for ic in item_colors):
                    return EligibilityResult(
                        passes=False,
                        reason=f"excluded_color:{ec}",
                        failed_rules=["color_exclusion"],
                    )
                # Check color_family
                if color_family and ec_lower in color_family:
                    return EligibilityResult(
                        passes=False,
                        reason=f"excluded_color_family:{ec}",
                        failed_rules=["color_exclusion"],
                    )

        # Check 8b: Rise inclusion
        if include_rise:
            item_rise = (getattr(candidate, "rise", None) or "").lower()
            if item_rise and item_rise not in {r.lower() for r in include_rise}:
                return EligibilityResult(
                    passes=False,
                    reason=f"rise_not_included:{item_rise}",
                    failed_rules=["rise_inclusion"],
                )

        # Check 8b: Rise exclusion
        if exclude_rise:
            item_rise = (getattr(candidate, "rise", None) or "").lower()
            if item_rise and item_rise in {r.lower() for r in exclude_rise}:
                return EligibilityResult(
                    passes=False,
                    reason=f"excluded_rise:{item_rise}",
                    failed_rules=["rise_exclusion"],
                )

        # Check 9: Unknown article type penalty (soft — passes but penalized)
        if canonical_type not in ARTICLE_TYPE_CANON.values():
            # Unknown type gets a penalty but still passes
            penalty = 0.2

        return EligibilityResult(
            passes=True,
            reason=None,
            penalty=penalty,
            failed_rules=failed_rules,
        )

    def filter(
        self,
        candidates: List[Candidate],
        occasions: Optional[List[str]] = None,
        user_exclusions: Optional[List[str]] = None,
        excluded_article_types: Optional[List[str]] = None,
        exclude_colors: Optional[List[str]] = None,
        exclude_brands: Optional[List[str]] = None,
        include_brands: Optional[List[str]] = None,
        include_rise: Optional[List[str]] = None,
        exclude_rise: Optional[List[str]] = None,
        hidden_ids: Optional[Set[str]] = None,
        negative_brands: Optional[Set[str]] = None,
        shown_set: Optional[Set[str]] = None,
    ) -> Tuple[List[Candidate], Dict[str, Any]]:
        """
        Filter a batch of candidates. Returns (passed, stats).

        Passed candidates have their penalty stored on the result but
        the candidate itself is not modified. The ranker reads the
        penalty from the filter result if needed.
        """
        passed = []
        penalties: Dict[str, float] = {}
        block_reasons: Dict[str, int] = {}

        for c in candidates:
            result = self.check(
                c,
                occasions=occasions,
                user_exclusions=user_exclusions,
                excluded_article_types=excluded_article_types,
                exclude_colors=exclude_colors,
                exclude_brands=exclude_brands,
                include_brands=include_brands,
                include_rise=include_rise,
                exclude_rise=exclude_rise,
                hidden_ids=hidden_ids,
                negative_brands=negative_brands,
                shown_set=shown_set,
            )
            if result.passes:
                passed.append(c)
                if result.penalty > 0:
                    penalties[c.item_id] = result.penalty
            else:
                reason = result.reason or "unknown"
                block_reasons[reason] = block_reasons.get(reason, 0) + 1

        logger.info(
            "Eligibility: %d/%d passed (%d blocked)",
            len(passed), len(candidates), len(candidates) - len(passed),
        )

        return passed, {"penalties": penalties, "block_reasons": block_reasons}
