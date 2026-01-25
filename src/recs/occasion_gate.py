"""
Occasion Gate Module

Multi-signal gating for accurate occasion-based filtering.

Approach:
1. Positive occasion score must be above threshold
2. Negative occasion score must be below threshold
3. Cross-occasion scores are checked (e.g., high "active" blocks "office")
4. Style scores are checked (e.g., high "sleeveless" may block "office")
5. Computed sleeve attribute provides hard gating

This solves the "is it appropriate for X?" question instead of "how X-y is it?".

Usage:
    from recs.occasion_gate import check_occasion_gate, OCCASION_GATES

    passes, reason = check_occasion_gate(candidate, ['office'])
    if not passes:
        print(f"Blocked: {reason}")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# Import Candidate type for type hints (avoid circular import)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from recs.models import Candidate


# =============================================================================
# Gate Configuration
# =============================================================================

@dataclass
class OccasionGateConfig:
    """
    Per-occasion gating configuration.

    Uses multi-signal approach:
    - Positive score: Item must score above this for the occasion
    - Negative score: Item must score below this (what the occasion is NOT)
    - Cross-occasion: Other occasion scores that must be below threshold
    - Style scores: Style scores that must be below threshold
    - Blocked sleeves: Sleeve values that disqualify the item
    """
    min_positive_score: float
    max_negative_score: float
    max_cross_occasion_scores: Dict[str, float] = field(default_factory=dict)
    max_style_scores: Dict[str, float] = field(default_factory=dict)
    blocked_sleeve_values: Set[str] = field(default_factory=set)
    blocked_length_values: Set[str] = field(default_factory=set)

    # Additional metadata-based blockers
    blocked_article_type_keywords: Set[str] = field(default_factory=set)
    blocked_brand_keywords: Set[str] = field(default_factory=set)


# =============================================================================
# Occasion Gate Definitions
# =============================================================================

OCCASION_GATES: Dict[str, OccasionGateConfig] = {
    'office': OccasionGateConfig(
        # Positive score threshold (item must be "office-y" enough)
        min_positive_score=0.18,
        # Negative score threshold (item must NOT be "anti-office")
        max_negative_score=0.30,
        # Cross-occasion blockers (high score in these = not office)
        max_cross_occasion_scores={
            'active': 0.28,    # Athletic items not office appropriate
            'beach': 0.25,     # Beach wear not office appropriate
        },
        # Style score blockers - use CLIP scores instead of hard sleeve block
        # This allows professional sleeveless items (shells, structured blouses)
        # while blocking casual tank tops via the active/sleeveless style scores
        max_style_scores={
            'sleeveless': 0.55,   # High sleeveless STYLE score = casual tank (not a shell/blouse)
            'crop-tops': 0.28,    # Crop tops not office appropriate
            'cutouts': 0.25,      # Cutouts not office appropriate
            'deep-necklines': 0.30,  # Deep necklines less office appropriate
            'sheer': 0.28,        # Sheer items not office appropriate
            'mini-lengths': 0.30, # Mini lengths less office appropriate
        },
        # NO hard sleeve blockers - rely on CLIP scores to distinguish
        # professional sleeveless (high office, low active) from casual tanks
        blocked_sleeve_values=set(),  # Removed: professional shells/blouses are sleeveless
        # Hard length blockers
        blocked_length_values={'mini'},  # Mini length hard blocked for office
        # Article type keyword blockers - be specific to avoid blocking professional items
        blocked_article_type_keywords={
            'tank top', 'racerback', 'bralette', 'sports bra',
            'bikini', 'swimsuit', 'legging', 'yoga pant', 'athletic',
            'gym', 'workout', 'running', 'jogger', 'sweatpant',
        },
        # Brand keyword blockers (athletic brands)
        blocked_brand_keywords={
            'alo', 'lululemon', 'athleta', 'fabletics', 'nike', 'adidas',
            'puma', 'under armour', 'reebok', 'gymshark', 'sweaty betty',
            'outdoor voices', 'beyond yoga', 'vuori', 'girlfriend collective',
        },
    ),

    'smart-casual': OccasionGateConfig(
        min_positive_score=0.16,
        max_negative_score=0.32,
        max_cross_occasion_scores={
            'active': 0.32,    # More lenient than office
            'beach': 0.28,
        },
        max_style_scores={
            'crop-tops': 0.30,
            'cutouts': 0.28,
            'sheer': 0.30,
        },
        blocked_sleeve_values=set(),  # Sleeveless OK for smart-casual
        blocked_length_values=set(),
        blocked_article_type_keywords={
            'sports bra', 'bikini', 'swimsuit', 'gym', 'workout',
        },
        blocked_brand_keywords=set(),  # Athletic brands may be OK for smart-casual
    ),

    'casual': OccasionGateConfig(
        # Very lenient - most things are casual
        min_positive_score=0.12,
        max_negative_score=0.40,
        max_cross_occasion_scores={},  # No cross-occasion blocks for casual
        max_style_scores={},  # No style blocks for casual
        blocked_sleeve_values=set(),
        blocked_length_values=set(),
        blocked_article_type_keywords=set(),
        blocked_brand_keywords=set(),
    ),

    'evening': OccasionGateConfig(
        min_positive_score=0.16,
        max_negative_score=0.35,
        max_cross_occasion_scores={
            'office': 0.32,   # Too office-y = not evening
            'active': 0.28,   # Athletic = not evening
        },
        max_style_scores={},  # Revealing styles often OK for evening
        blocked_sleeve_values=set(),
        blocked_length_values=set(),
        blocked_article_type_keywords={
            'sports bra', 'gym', 'workout', 'running', 'athletic',
            'office', 'business', 'work',  # Too professional for evening
        },
        blocked_brand_keywords=set(),
    ),

    'events': OccasionGateConfig(
        min_positive_score=0.16,
        max_negative_score=0.35,
        max_cross_occasion_scores={
            'active': 0.28,
        },
        max_style_scores={},  # More permissive for special events
        blocked_sleeve_values=set(),
        blocked_length_values=set(),
        blocked_article_type_keywords={
            'sports bra', 'gym', 'workout', 'running', 'athletic',
        },
        blocked_brand_keywords=set(),
    ),

    'beach': OccasionGateConfig(
        min_positive_score=0.16,
        max_negative_score=0.35,
        max_cross_occasion_scores={
            'office': 0.30,   # Too office-y = not beach
        },
        max_style_scores={},  # Revealing styles OK for beach
        blocked_sleeve_values=set(),
        blocked_length_values=set(),
        blocked_article_type_keywords={
            'suit', 'blazer', 'formal', 'business',
        },
        blocked_brand_keywords=set(),
    ),

    'active': OccasionGateConfig(
        # Must clearly be athletic
        min_positive_score=0.22,
        max_negative_score=0.30,
        max_cross_occasion_scores={
            'office': 0.28,   # Too office-y = not athletic
            'evening': 0.28,  # Too dressy = not athletic
        },
        max_style_scores={
            'sheer': 0.30,    # Sheer not athletic
        },
        blocked_sleeve_values=set(),
        blocked_length_values=set(),
        blocked_article_type_keywords={
            'formal', 'suit', 'blazer', 'cocktail', 'evening gown',
        },
        blocked_brand_keywords=set(),
    ),
}


# =============================================================================
# Gate Checking Function
# =============================================================================

def check_occasion_gate(
    candidate: "Candidate",
    occasions: List[str],
    verbose: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Check if candidate passes occasion gate using multi-signal approach.

    Args:
        candidate: Candidate item with computed scores
        occasions: List of occasions to check (item must pass ALL)
        verbose: If True, print detailed gate checks

    Returns:
        (passes, reason) - reason explains why it failed (None if passes)
    """
    if not occasions:
        return True, None

    # Get candidate's computed scores
    occasion_scores = candidate.computed_occasion_scores or {}
    style_scores = candidate.computed_style_scores or {}

    # Get candidate metadata
    name_lower = (candidate.name or '').lower()
    brand_lower = (candidate.brand or '').lower()
    article_type_lower = (candidate.article_type or '').lower()
    sleeve = (candidate.sleeve or '').lower()
    length = (candidate.length or '').lower()

    # Determine if we have computed scores available
    # If not, we'll only use hard blockers (brand, keywords, sleeve, length)
    has_occasion_scores = bool(occasion_scores)
    has_style_scores = bool(style_scores)

    for occasion in occasions:
        occasion_lower = occasion.lower()
        gate = OCCASION_GATES.get(occasion_lower)

        if not gate:
            # Unknown occasion - pass by default
            if verbose:
                print(f"[OccasionGate] Unknown occasion '{occasion}', passing")
            continue

        # Score-based checks (only if scores are available)
        if has_occasion_scores:
            # Check 1: Positive score threshold
            positive_score = occasion_scores.get(occasion_lower, 0)
            if positive_score < gate.min_positive_score:
                reason = f"low {occasion} score ({positive_score:.2f} < {gate.min_positive_score})"
                if verbose:
                    print(f"[OccasionGate] BLOCKED: {reason}")
                return False, reason

            # Check 2: Negative score threshold
            negative_key = f"{occasion_lower}_negative"
            negative_score = occasion_scores.get(negative_key, 0)
            if negative_score > gate.max_negative_score:
                reason = f"high {occasion} negative score ({negative_score:.2f} > {gate.max_negative_score})"
                if verbose:
                    print(f"[OccasionGate] BLOCKED: {reason}")
                return False, reason

            # Check 3: Cross-occasion scores
            for cross_occ, max_val in gate.max_cross_occasion_scores.items():
                cross_score = occasion_scores.get(cross_occ, 0)
                if cross_score > max_val:
                    reason = f"high {cross_occ} score ({cross_score:.2f} > {max_val}) blocks {occasion}"
                    if verbose:
                        print(f"[OccasionGate] BLOCKED: {reason}")
                    return False, reason

        # Check 4: Style scores (only if style scores are available)
        if has_style_scores:
            for style, max_val in gate.max_style_scores.items():
                style_score = style_scores.get(style, 0)
                if style_score > max_val:
                    reason = f"high {style} style ({style_score:.2f} > {max_val}) blocks {occasion}"
                    if verbose:
                        print(f"[OccasionGate] BLOCKED: {reason}")
                    return False, reason

        # Hard blockers (always applied regardless of scores availability)
        # Check 5: Hard sleeve blockers
        if sleeve in gate.blocked_sleeve_values:
            reason = f"sleeve type '{sleeve}' blocked for {occasion}"
            if verbose:
                print(f"[OccasionGate] BLOCKED: {reason}")
            return False, reason

        # Check 6: Hard length blockers
        if length in gate.blocked_length_values:
            reason = f"length '{length}' blocked for {occasion}"
            if verbose:
                print(f"[OccasionGate] BLOCKED: {reason}")
            return False, reason

        # Check 7: Article type keyword blockers
        combined_text = f"{name_lower} {article_type_lower}"
        for keyword in gate.blocked_article_type_keywords:
            if keyword in combined_text:
                reason = f"article type keyword '{keyword}' blocked for {occasion}"
                if verbose:
                    print(f"[OccasionGate] BLOCKED: {reason}")
                return False, reason

        # Check 8: Brand keyword blockers
        for keyword in gate.blocked_brand_keywords:
            if keyword in brand_lower:
                reason = f"brand '{keyword}' blocked for {occasion}"
                if verbose:
                    print(f"[OccasionGate] BLOCKED: {reason}")
                return False, reason

    # All checks passed
    if verbose:
        print(f"[OccasionGate] PASSED all checks for {occasions}")
    return True, None


def filter_candidates_by_occasion(
    candidates: List["Candidate"],
    occasions: List[str],
    verbose: bool = False,
) -> Tuple[List["Candidate"], Dict[str, int]]:
    """
    Filter a list of candidates by occasion gates.

    Args:
        candidates: List of candidates to filter
        occasions: List of occasions to check
        verbose: If True, print summary statistics

    Returns:
        (filtered_candidates, blocked_reasons) - blocked_reasons is a count by reason
    """
    from collections import defaultdict

    filtered = []
    blocked_reasons = defaultdict(int)

    for candidate in candidates:
        passes, reason = check_occasion_gate(candidate, occasions)
        if passes:
            filtered.append(candidate)
        else:
            blocked_reasons[reason] += 1

    if verbose and blocked_reasons:
        print(f"[OccasionGate] Filtered {len(candidates)} -> {len(filtered)} candidates")
        print(f"[OccasionGate] Blocked reasons:")
        for reason, count in sorted(blocked_reasons.items(), key=lambda x: -x[1])[:10]:
            print(f"  - {reason}: {count}")

    return filtered, dict(blocked_reasons)


# =============================================================================
# Testing
# =============================================================================

def test_occasion_gate():
    """Test occasion gate functionality."""
    from recs.models import Candidate

    print("=" * 70)
    print("Testing Occasion Gate")
    print("=" * 70)

    # Test 1: Office-appropriate blouse
    print("\n1. Testing office-appropriate blouse...")
    blouse = Candidate(
        item_id="test-blouse-1",
        name="Professional Silk Blouse",
        brand="Theory",
        article_type="Blouse",
        sleeve="long-sleeve",
        length="standard",
        computed_occasion_scores={
            'office': 0.35,
            'office_negative': 0.15,
            'casual': 0.28,
            'active': 0.12,
            'beach': 0.10,
        },
        computed_style_scores={
            'sleeveless': 0.08,
            'crop-tops': 0.05,
            'cutouts': 0.02,
            'deep-necklines': 0.10,
            'sheer': 0.15,
        },
    )
    passes, reason = check_occasion_gate(blouse, ['office'], verbose=True)
    print(f"   Result: {'PASS' if passes else f'FAIL - {reason}'}")
    assert passes, f"Blouse should pass office gate: {reason}"

    # Test 2: Athletic tank top - should NOT pass office
    print("\n2. Testing athletic tank top (should fail office)...")
    tank = Candidate(
        item_id="test-tank-1",
        name="Alo Yoga Airbrush Tank",
        brand="Alo Yoga",
        article_type="Tank Top",
        sleeve="sleeveless",
        length="standard",
        computed_occasion_scores={
            'office': 0.19,  # Above threshold but...
            'office_negative': 0.38,  # High negative score
            'casual': 0.42,
            'active': 0.45,  # High active score
            'beach': 0.35,
        },
        computed_style_scores={
            'sleeveless': 0.85,  # Very sleeveless
            'crop-tops': 0.20,
            'cutouts': 0.05,
        },
    )
    passes, reason = check_occasion_gate(tank, ['office'], verbose=True)
    print(f"   Result: {'PASS' if passes else f'FAIL - {reason}'}")
    assert not passes, "Athletic tank should NOT pass office gate"

    # Test 3: Crop top - should NOT pass office (style score)
    print("\n3. Testing crop top (should fail office)...")
    crop = Candidate(
        item_id="test-crop-1",
        name="Cute Crop Top",
        brand="Zara",
        article_type="Crop Top",
        sleeve="short-sleeve",
        length="cropped",
        computed_occasion_scores={
            'office': 0.22,
            'office_negative': 0.25,
            'casual': 0.45,
            'active': 0.20,
        },
        computed_style_scores={
            'sleeveless': 0.15,
            'crop-tops': 0.75,  # Very crop-top-y
            'cutouts': 0.10,
        },
    )
    passes, reason = check_occasion_gate(crop, ['office'], verbose=True)
    print(f"   Result: {'PASS' if passes else f'FAIL - {reason}'}")
    assert not passes, "Crop top should NOT pass office gate"

    # Test 4: Sleeveless dress should fail office (hard sleeve block)
    print("\n4. Testing sleeveless dress (should fail office due to sleeve)...")
    sleeveless_dress = Candidate(
        item_id="test-dress-1",
        name="Summer Sleeveless Dress",
        brand="Reformation",
        article_type="Dress",
        sleeve="sleeveless",  # Hard blocked
        length="midi",
        computed_occasion_scores={
            'office': 0.28,
            'office_negative': 0.18,
            'casual': 0.35,
            'active': 0.10,
        },
        computed_style_scores={
            'sleeveless': 0.70,
            'crop-tops': 0.05,
        },
    )
    passes, reason = check_occasion_gate(sleeveless_dress, ['office'], verbose=True)
    print(f"   Result: {'PASS' if passes else f'FAIL - {reason}'}")
    assert not passes, "Sleeveless dress should NOT pass office gate"

    # Test 5: Same dress should pass casual
    print("\n5. Testing same sleeveless dress (should pass casual)...")
    passes, reason = check_occasion_gate(sleeveless_dress, ['casual'], verbose=True)
    print(f"   Result: {'PASS' if passes else f'FAIL - {reason}'}")
    assert passes, f"Sleeveless dress should pass casual gate: {reason}"

    # Test 6: Athletic brand item should fail office
    print("\n6. Testing Lululemon pullover (should fail office due to brand)...")
    lulu = Candidate(
        item_id="test-lulu-1",
        name="Scuba Full Zip Hoodie",
        brand="Lululemon",
        article_type="Hoodie",
        sleeve="long-sleeve",
        length="standard",
        computed_occasion_scores={
            'office': 0.20,
            'office_negative': 0.32,
            'casual': 0.40,
            'active': 0.38,
        },
        computed_style_scores={
            'sleeveless': 0.05,
            'crop-tops': 0.08,
        },
    )
    passes, reason = check_occasion_gate(lulu, ['office'], verbose=True)
    print(f"   Result: {'PASS' if passes else f'FAIL - {reason}'}")
    assert not passes, "Lululemon hoodie should NOT pass office gate"

    # Test 7: Test batch filtering
    print("\n7. Testing batch filtering...")
    candidates = [blouse, tank, crop, sleeveless_dress, lulu]
    filtered, blocked_reasons = filter_candidates_by_occasion(candidates, ['office'], verbose=True)
    print(f"   Filtered {len(candidates)} -> {len(filtered)} items for office")
    assert len(filtered) == 1, f"Should have 1 office-appropriate item, got {len(filtered)}"

    print("\n" + "=" * 70)
    print("Occasion Gate tests complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_occasion_gate()
