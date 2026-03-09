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
        # Extended PA attribute filters
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        exclude_styles: Optional[List[str]] = None,
        include_fit: Optional[List[str]] = None,
        exclude_fit: Optional[List[str]] = None,
        include_length: Optional[List[str]] = None,
        exclude_length: Optional[List[str]] = None,
        include_sleeves: Optional[List[str]] = None,
        exclude_sleeves: Optional[List[str]] = None,
        include_neckline: Optional[List[str]] = None,
        exclude_neckline: Optional[List[str]] = None,
        include_formality: Optional[List[str]] = None,
        exclude_formality: Optional[List[str]] = None,
        include_seasons: Optional[List[str]] = None,
        exclude_seasons: Optional[List[str]] = None,
        include_silhouette: Optional[List[str]] = None,
        exclude_silhouette: Optional[List[str]] = None,
        include_color_family: Optional[List[str]] = None,
        exclude_color_family: Optional[List[str]] = None,
        include_style_tags: Optional[List[str]] = None,
        exclude_style_tags: Optional[List[str]] = None,
        include_coverage: Optional[List[str]] = None,
        exclude_coverage: Optional[List[str]] = None,
        include_materials: Optional[List[str]] = None,
        exclude_materials: Optional[List[str]] = None,
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
                # OCCASION_ALLOWED is Dict[occasion, Dict[broad_category, Set[types]]]
                # Flatten to a single set of all allowed types for this occasion
                allowed_nested = OCCASION_ALLOWED.get(occ_lower)
                if allowed_nested is not None:
                    all_allowed: set = set()
                    for type_set in allowed_nested.values():
                        all_allowed.update(type_set)
                    if canonical_type not in all_allowed:
                        # Check by name too
                        if not any(a in name_lower for a in all_allowed if a):
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

        # ---------------------------------------------------------------
        # Check 10: Pattern filters (include/exclude)
        # ---------------------------------------------------------------
        item_pattern = (getattr(candidate, "pattern", None) or "").lower()

        if include_patterns and item_pattern:
            ip_lower = {p.lower() for p in include_patterns}
            if item_pattern not in ip_lower:
                return EligibilityResult(
                    passes=False,
                    reason=f"pattern_not_included:{item_pattern}",
                    failed_rules=["pattern_inclusion"],
                )

        if exclude_patterns and item_pattern:
            ep_lower = {p.lower() for p in exclude_patterns}
            if item_pattern in ep_lower:
                return EligibilityResult(
                    passes=False,
                    reason=f"excluded_pattern:{item_pattern}",
                    failed_rules=["pattern_exclusion"],
                )

        # ---------------------------------------------------------------
        # Check 11: Coverage/style exclusion (sheer, cutouts, backless, etc.)
        # ---------------------------------------------------------------
        if exclude_styles:
            es_lower = {s.lower().replace("-", "_").replace(" ", "_") for s in exclude_styles}
            # Check coverage_details array
            coverage_details = getattr(candidate, "coverage_details", None) or []
            for cd in coverage_details:
                cd_norm = cd.lower().replace("-", "_").replace(" ", "_")
                if cd_norm in es_lower:
                    return EligibilityResult(
                        passes=False,
                        reason=f"excluded_style:{cd}",
                        failed_rules=["style_exclusion"],
                    )
            # Also check neckline for deep-necklines
            neckline = (getattr(candidate, "neckline", None) or "").lower()
            if "deep_necklines" in es_lower or "deep-necklines" in es_lower:
                if neckline and any(d in neckline for d in ("plunging", "deep v", "deep-v")):
                    return EligibilityResult(
                        passes=False,
                        reason=f"excluded_style:deep_neckline:{neckline}",
                        failed_rules=["style_exclusion"],
                    )

        # ---------------------------------------------------------------
        # Check 12+: Extended PA attribute filters (soft_preferences)
        # Each is an include/exclude pair matching a Candidate field.
        # ---------------------------------------------------------------

        # Fit
        item_fit = (getattr(candidate, "fit", None) or "").lower()
        if include_fit and item_fit:
            if item_fit not in {f.lower() for f in include_fit}:
                return EligibilityResult(
                    passes=False,
                    reason=f"fit_not_included:{item_fit}",
                    failed_rules=["fit_inclusion"],
                )
        if exclude_fit and item_fit:
            if item_fit in {f.lower() for f in exclude_fit}:
                return EligibilityResult(
                    passes=False,
                    reason=f"excluded_fit:{item_fit}",
                    failed_rules=["fit_exclusion"],
                )

        # Length
        item_length = (getattr(candidate, "length", None) or "").lower()
        if include_length and item_length:
            if item_length not in {l.lower() for l in include_length}:
                return EligibilityResult(
                    passes=False,
                    reason=f"length_not_included:{item_length}",
                    failed_rules=["length_inclusion"],
                )
        if exclude_length and item_length:
            if item_length in {l.lower() for l in exclude_length}:
                return EligibilityResult(
                    passes=False,
                    reason=f"excluded_length:{item_length}",
                    failed_rules=["length_exclusion"],
                )

        # Sleeves
        item_sleeve = (getattr(candidate, "sleeve", None) or "").lower()
        if include_sleeves and item_sleeve:
            if item_sleeve not in {s.lower() for s in include_sleeves}:
                return EligibilityResult(
                    passes=False,
                    reason=f"sleeve_not_included:{item_sleeve}",
                    failed_rules=["sleeve_inclusion"],
                )
        if exclude_sleeves and item_sleeve:
            if item_sleeve in {s.lower() for s in exclude_sleeves}:
                return EligibilityResult(
                    passes=False,
                    reason=f"excluded_sleeve:{item_sleeve}",
                    failed_rules=["sleeve_exclusion"],
                )

        # Neckline
        item_neckline = (getattr(candidate, "neckline", None) or "").lower()
        if include_neckline and item_neckline:
            if item_neckline not in {n.lower() for n in include_neckline}:
                return EligibilityResult(
                    passes=False,
                    reason=f"neckline_not_included:{item_neckline}",
                    failed_rules=["neckline_inclusion"],
                )
        if exclude_neckline and item_neckline:
            if item_neckline in {n.lower() for n in exclude_neckline}:
                return EligibilityResult(
                    passes=False,
                    reason=f"excluded_neckline:{item_neckline}",
                    failed_rules=["neckline_exclusion"],
                )

        # Formality
        item_formality = (getattr(candidate, "formality", None) or "").lower()
        if include_formality and item_formality:
            if item_formality not in {f.lower() for f in include_formality}:
                return EligibilityResult(
                    passes=False,
                    reason=f"formality_not_included:{item_formality}",
                    failed_rules=["formality_inclusion"],
                )
        if exclude_formality and item_formality:
            if item_formality in {f.lower() for f in exclude_formality}:
                return EligibilityResult(
                    passes=False,
                    reason=f"excluded_formality:{item_formality}",
                    failed_rules=["formality_exclusion"],
                )

        # Seasons (list field)
        item_seasons = [s.lower() for s in (getattr(candidate, "seasons", None) or [])]
        if include_seasons and item_seasons:
            is_lower = {s.lower() for s in include_seasons}
            if not any(s in is_lower for s in item_seasons):
                return EligibilityResult(
                    passes=False,
                    reason=f"season_not_included:{item_seasons}",
                    failed_rules=["season_inclusion"],
                )
        if exclude_seasons and item_seasons:
            es_lower = {s.lower() for s in exclude_seasons}
            if any(s in es_lower for s in item_seasons):
                return EligibilityResult(
                    passes=False,
                    reason=f"excluded_season:{item_seasons}",
                    failed_rules=["season_exclusion"],
                )

        # Silhouette
        item_silhouette = (getattr(candidate, "silhouette", None) or "").lower()
        if include_silhouette and item_silhouette:
            if item_silhouette not in {s.lower() for s in include_silhouette}:
                return EligibilityResult(
                    passes=False,
                    reason=f"silhouette_not_included:{item_silhouette}",
                    failed_rules=["silhouette_inclusion"],
                )
        if exclude_silhouette and item_silhouette:
            if item_silhouette in {s.lower() for s in exclude_silhouette}:
                return EligibilityResult(
                    passes=False,
                    reason=f"excluded_silhouette:{item_silhouette}",
                    failed_rules=["silhouette_exclusion"],
                )

        # Color family
        item_cf = (getattr(candidate, "color_family", None) or "").lower()
        if include_color_family and item_cf:
            if item_cf not in {c.lower() for c in include_color_family}:
                return EligibilityResult(
                    passes=False,
                    reason=f"color_family_not_included:{item_cf}",
                    failed_rules=["color_family_inclusion"],
                )
        if exclude_color_family and item_cf:
            if item_cf in {c.lower() for c in exclude_color_family}:
                return EligibilityResult(
                    passes=False,
                    reason=f"excluded_color_family:{item_cf}",
                    failed_rules=["color_family_exclusion"],
                )

        # Style tags (list field)
        item_tags = [t.lower() for t in (getattr(candidate, "style_tags", None) or [])]
        if include_style_tags and item_tags:
            ist_lower = {t.lower() for t in include_style_tags}
            if not any(t in ist_lower for t in item_tags):
                return EligibilityResult(
                    passes=False,
                    reason=f"style_tag_not_included:{item_tags}",
                    failed_rules=["style_tag_inclusion"],
                )
        if exclude_style_tags and item_tags:
            est_lower = {t.lower() for t in exclude_style_tags}
            if any(t in est_lower for t in item_tags):
                return EligibilityResult(
                    passes=False,
                    reason=f"excluded_style_tag:{item_tags}",
                    failed_rules=["style_tag_exclusion"],
                )

        # Coverage level
        item_coverage = (getattr(candidate, "coverage_level", None) or "").lower()
        if include_coverage and item_coverage:
            if item_coverage not in {c.lower() for c in include_coverage}:
                return EligibilityResult(
                    passes=False,
                    reason=f"coverage_not_included:{item_coverage}",
                    failed_rules=["coverage_inclusion"],
                )
        if exclude_coverage and item_coverage:
            if item_coverage in {c.lower() for c in exclude_coverage}:
                return EligibilityResult(
                    passes=False,
                    reason=f"excluded_coverage:{item_coverage}",
                    failed_rules=["coverage_exclusion"],
                )

        # Materials (list field)
        item_materials = [m.lower() for m in (getattr(candidate, "materials", None) or [])]
        if include_materials and item_materials:
            im_lower = {m.lower() for m in include_materials}
            if not any(m in im_lower for m in item_materials):
                return EligibilityResult(
                    passes=False,
                    reason=f"material_not_included:{item_materials}",
                    failed_rules=["material_inclusion"],
                )
        if exclude_materials and item_materials:
            em_lower = {m.lower() for m in exclude_materials}
            if any(m in em_lower for m in item_materials):
                return EligibilityResult(
                    passes=False,
                    reason=f"excluded_material:{item_materials}",
                    failed_rules=["material_exclusion"],
                )

        # ---------------------------------------------------------------
        # Check 9: Unknown article type penalty (soft — passes but penalized)
        # ---------------------------------------------------------------
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
        **kwargs,
    ) -> Tuple[List[Candidate], Dict[str, Any]]:
        """
        Filter a batch of candidates. Returns (passed, stats).

        All keyword arguments are forwarded to ``check()``.
        Accepts all core filters (occasions, exclude_brands, include_brands, etc.)
        plus all extended PA attribute filters (include_patterns, exclude_fit, etc.).
        """
        passed = []
        penalties: Dict[str, float] = {}
        block_reasons: Dict[str, int] = {}

        for c in candidates:
            result = self.check(c, **kwargs)
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
