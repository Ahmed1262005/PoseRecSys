#!/usr/bin/env python3
"""
Test Script: Preference Weighting and Brand Prioritization

Tests:
1. Brand boost effectiveness at different brand_openness levels
2. Cold start mitigation with onboarding preferences
3. Type preference filtering
4. Comparison of scores with/without preferences

Run: cd /home/ubuntu/recSys/outfitTransformer/src && python3 -m recs.test_preference_weighting
"""

import os
import sys
from collections import defaultdict
from typing import Dict, List, Any
import json

# Add src directory to path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.dirname(src_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, root_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(root_dir, '.env'))

from recs.candidate_selection import CandidateSelectionModule, CandidateSelectionConfig
from recs.models import (
    UserState, UserStateType, OnboardingProfile,
    TopsPrefs, BottomsPrefs, DressesPrefs, OuterwearPrefs
)
from recs.pipeline import RecommendationPipeline


def print_separator(title: str = ""):
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def analyze_brand_distribution(candidates: List, preferred_brands: List[str]) -> Dict:
    """Analyze brand distribution in candidates."""
    brand_counts = defaultdict(int)
    preferred_count = 0
    preferred_brands_lower = set(b.lower() for b in preferred_brands)

    for c in candidates:
        brand = (c.brand or "Unknown").strip()
        brand_counts[brand] += 1
        if brand.lower() in preferred_brands_lower:
            preferred_count += 1

    # Top 10 brands
    top_brands = sorted(brand_counts.items(), key=lambda x: -x[1])[:10]

    return {
        "total": len(candidates),
        "preferred_count": preferred_count,
        "preferred_percent": round(preferred_count / len(candidates) * 100, 1) if candidates else 0,
        "top_brands": top_brands,
        "unique_brands": len(brand_counts)
    }


def analyze_scores(candidates: List) -> Dict:
    """Analyze score distribution."""
    if not candidates:
        return {}

    emb_scores = [c.embedding_score for c in candidates]
    pref_scores = [c.preference_score for c in candidates]
    final_scores = [c.final_score for c in candidates]

    return {
        "embedding": {
            "min": round(min(emb_scores), 3),
            "max": round(max(emb_scores), 3),
            "avg": round(sum(emb_scores) / len(emb_scores), 3)
        },
        "preference": {
            "min": round(min(pref_scores), 3),
            "max": round(max(pref_scores), 3),
            "avg": round(sum(pref_scores) / len(pref_scores), 3)
        },
        "final": {
            "min": round(min(final_scores), 3),
            "max": round(max(final_scores), 3),
            "avg": round(sum(final_scores) / len(final_scores), 3)
        }
    }


def test_brand_boost_levels():
    """Test brand boost at different brand_openness levels."""
    print_separator("TEST 1: Brand Boost at Different Openness Levels")

    module = CandidateSelectionModule()

    # Define test brands (common women's fashion brands)
    test_preferred_brands = ["Zara", "H&M", "Mango", "ASOS"]

    openness_levels = [
        ("stick_to_favorites", "2.0x brand boost"),
        ("mix", "1.0x brand boost"),
        ("mix-favorites-new", "0.8x brand boost"),
        ("discover_new", "0.5x brand boost"),
    ]

    results = {}

    for openness, description in openness_levels:
        print(f"\n--- Brand Openness: {openness} ({description}) ---")

        # Create user state with onboarding
        profile = OnboardingProfile(
            user_id="test_brand_user",
            categories=["tops", "dresses"],
            preferred_brands=test_preferred_brands,
            brand_openness=openness,
            style_directions=["minimal", "classic"]
        )

        state = UserState(
            user_id="test_brand_user",
            state_type=UserStateType.TINDER_COMPLETE,
            onboarding_profile=profile
        )

        # Get candidates
        candidates = module.get_candidates(state, gender="female")

        # Analyze
        brand_analysis = analyze_brand_distribution(candidates, test_preferred_brands)
        score_analysis = analyze_scores(candidates)

        print(f"  Total candidates: {brand_analysis['total']}")
        print(f"  Preferred brand items: {brand_analysis['preferred_count']} ({brand_analysis['preferred_percent']}%)")
        print(f"  Unique brands: {brand_analysis['unique_brands']}")
        print(f"  Preference score avg: {score_analysis.get('preference', {}).get('avg', 0)}")

        # Show top brands
        print(f"  Top 5 brands:")
        for brand, count in brand_analysis['top_brands'][:5]:
            is_preferred = "★" if brand.lower() in [b.lower() for b in test_preferred_brands] else " "
            print(f"    {is_preferred} {brand}: {count}")

        # Show top 5 by final score
        print(f"  Top 5 by final score:")
        sorted_candidates = sorted(candidates, key=lambda x: x.final_score, reverse=True)[:5]
        for i, c in enumerate(sorted_candidates, 1):
            is_preferred = "★" if (c.brand or "").lower() in [b.lower() for b in test_preferred_brands] else " "
            print(f"    {i}. {is_preferred} {c.brand or 'N/A'} | score={c.final_score:.3f} pref={c.preference_score:.3f}")

        results[openness] = {
            "preferred_percent": brand_analysis['preferred_percent'],
            "avg_preference_score": score_analysis.get('preference', {}).get('avg', 0),
            "preferred_in_top10": sum(1 for c in sorted_candidates[:10]
                                     if (c.brand or "").lower() in [b.lower() for b in test_preferred_brands])
        }

    # Summary
    print("\n--- Summary ---")
    print(f"{'Openness':<25} {'Preferred %':<15} {'Avg Pref Score':<15} {'In Top 10':<10}")
    print("-" * 65)
    for openness, data in results.items():
        print(f"{openness:<25} {data['preferred_percent']:<15.1f} {data['avg_preference_score']:<15.3f} {data['preferred_in_top10']:<10}")

    return results


def test_cold_start_with_onboarding():
    """Test cold start mitigation using onboarding preferences."""
    print_separator("TEST 2: Cold Start Mitigation with Onboarding")

    module = CandidateSelectionModule()

    # Test 1: Pure cold start (no onboarding)
    print("\n--- Scenario A: Pure Cold Start (no onboarding) ---")
    cold_state = UserState(
        user_id="test_pure_cold",
        state_type=UserStateType.COLD_START
    )

    cold_candidates = module.get_candidates(cold_state, gender="female")
    cold_analysis = analyze_scores(cold_candidates)

    print(f"  Total candidates: {len(cold_candidates)}")
    print(f"  Avg preference score: {cold_analysis.get('preference', {}).get('avg', 0)}")

    by_source = defaultdict(int)
    for c in cold_candidates:
        by_source[c.source] += 1
    print(f"  By source: {dict(by_source)}")

    # Test 2: Cold start WITH onboarding (no taste vector)
    print("\n--- Scenario B: Cold Start WITH Onboarding (no taste vector) ---")

    profile = OnboardingProfile(
        user_id="test_onboard_cold",
        categories=["tops", "dresses"],
        preferred_brands=["Zara", "H&M"],
        colors_to_avoid=["orange", "neon"],
        materials_to_avoid=["polyester"],
        style_directions=["minimal"],
        tops=TopsPrefs(
            types=["blouse", "tee"],
            fits=["regular", "relaxed"],
            sleeves=["short-sleeve", "long-sleeve"]
        ),
        dresses=DressesPrefs(
            types=["wrap-dress", "a-line-dress"],
            fits=["regular"],
            lengths=["midi", "maxi"]
        )
    )

    onboard_state = UserState(
        user_id="test_onboard_cold",
        state_type=UserStateType.TINDER_COMPLETE,  # Treated as tinder_complete due to onboarding
        onboarding_profile=profile
    )

    onboard_candidates = module.get_candidates(onboard_state, gender="female")
    onboard_analysis = analyze_scores(onboard_candidates)

    print(f"  Total candidates: {len(onboard_candidates)}")
    print(f"  Avg preference score: {onboard_analysis.get('preference', {}).get('avg', 0)}")

    by_source = defaultdict(int)
    for c in onboard_candidates:
        by_source[c.source] += 1
    print(f"  By source: {dict(by_source)}")

    # Check filter effectiveness
    filter_violations = 0
    for c in onboard_candidates:
        if any(color in ['orange', 'neon'] for color in (c.colors or [])):
            filter_violations += 1
        if 'polyester' in [m.lower() for m in (c.materials or [])]:
            filter_violations += 1
    print(f"  Filter violations (orange/neon/polyester): {filter_violations}")

    # Brand distribution
    brand_analysis = analyze_brand_distribution(onboard_candidates, ["Zara", "H&M"])
    print(f"  Preferred brand items: {brand_analysis['preferred_count']} ({brand_analysis['preferred_percent']}%)")

    # Compare top 5
    print("\n  Top 5 candidates (with onboarding):")
    sorted_onboard = sorted(onboard_candidates, key=lambda x: x.final_score, reverse=True)[:5]
    for i, c in enumerate(sorted_onboard, 1):
        print(f"    {i}. {c.brand or 'N/A':<15} | {c.category:<20} | final={c.final_score:.3f} pref={c.preference_score:.3f}")


def test_type_filtering():
    """Test type preference filtering."""
    print_separator("TEST 3: Type Preference Filtering")

    module = CandidateSelectionModule()

    # Profile with specific type preferences
    profile = OnboardingProfile(
        user_id="test_types",
        categories=["tops", "bottoms"],
        tops=TopsPrefs(
            types=["blouse", "sweater", "cardigan"],
            fits=["regular"]
        ),
        bottoms=BottomsPrefs(
            types=["jeans", "pants"],
            fits=["regular", "slim"],
            rises=["high-rise", "mid-rise"]
        )
    )

    state = UserState(
        user_id="test_types",
        state_type=UserStateType.TINDER_COMPLETE,
        onboarding_profile=profile
    )

    candidates = module.get_candidates(state, gender="female")

    # Analyze by article_type
    type_counts = defaultdict(int)
    type_scores = defaultdict(list)

    for c in candidates:
        article_type = c.article_type or c.category or "unknown"
        type_counts[article_type] += 1
        type_scores[article_type].append(c.preference_score)

    print(f"\nTotal candidates: {len(candidates)}")
    print(f"\nArticle type distribution:")
    sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])[:15]

    preferred_types = ["blouse", "sweater", "cardigan", "jeans", "pants"]

    for article_type, count in sorted_types:
        avg_score = sum(type_scores[article_type]) / len(type_scores[article_type])
        is_match = "★" if any(pt in article_type.lower() for pt in preferred_types) else " "
        print(f"  {is_match} {article_type:<25} count={count:<5} avg_pref_score={avg_score:.3f}")

    # Check if preferred types have higher scores
    preferred_scores = []
    other_scores = []
    for c in candidates:
        article_type = (c.article_type or c.category or "").lower()
        if any(pt in article_type for pt in preferred_types):
            preferred_scores.append(c.preference_score)
        else:
            other_scores.append(c.preference_score)

    if preferred_scores and other_scores:
        print(f"\n  Preferred type avg score: {sum(preferred_scores)/len(preferred_scores):.3f}")
        print(f"  Other type avg score: {sum(other_scores)/len(other_scores):.3f}")


def test_full_pipeline():
    """Test full pipeline with feed endpoint simulation."""
    print_separator("TEST 4: Full Pipeline Feed Generation")

    pipeline = RecommendationPipeline(load_sasrec=False)  # Skip SASRec for speed

    # Save a test profile first
    profile = OnboardingProfile(
        user_id="test_pipeline_feed",
        categories=["tops", "dresses", "bottoms"],
        preferred_brands=["Zara", "H&M", "Mango"],
        brands_to_avoid=["Shein"],
        brand_openness="mix-favorites-new",
        colors_to_avoid=["neon"],
        style_directions=["minimal", "classic"],
        tops=TopsPrefs(
            types=["blouse", "tee", "sweater"],
            fits=["regular", "relaxed"],
            sleeves=["short-sleeve", "long-sleeve"]
        ),
        dresses=DressesPrefs(
            types=["wrap-dress", "midi"],
            fits=["regular"],
            lengths=["midi", "maxi"]
        ),
        bottoms=BottomsPrefs(
            types=["jeans", "pants"],
            fits=["regular", "slim"],
            rises=["high-rise"]
        )
    )

    # Save profile
    print("\nSaving test profile...")
    save_result = pipeline.save_onboarding(profile, gender="female")
    print(f"  Save status: {save_result.get('status')}")

    # Test feed multiple times
    print("\nRunning feed generation 3 times...")

    all_results = []
    preferred_brands = ["zara", "h&m", "mango"]

    for run in range(3):
        print(f"\n--- Run {run + 1} ---")

        response = pipeline.get_feed_keyset(
            anon_id="test_pipeline_feed",
            gender="female",
            page_size=50
        )

        results = response.get("results", [])

        # Analyze
        brand_counts = defaultdict(int)
        preferred_count = 0

        for item in results:
            brand = (item.get("brand") or "Unknown").lower()
            brand_counts[brand] += 1
            if brand in preferred_brands:
                preferred_count += 1

        all_results.append({
            "total": len(results),
            "preferred_count": preferred_count,
            "strategy": response.get("strategy"),
            "top_brands": sorted(brand_counts.items(), key=lambda x: -x[1])[:5]
        })

        print(f"  Strategy: {response.get('strategy')}")
        print(f"  Total results: {len(results)}")
        print(f"  Preferred brands: {preferred_count} ({preferred_count/len(results)*100:.1f}%)" if results else "  No results")

        # Top 5 items
        print(f"  Top 5 items:")
        for i, item in enumerate(results[:5], 1):
            brand = item.get("brand", "N/A")
            is_preferred = "★" if brand.lower() in preferred_brands else " "
            print(f"    {i}. {is_preferred} {brand:<15} | {item.get('category', 'N/A'):<15} | score={item.get('score', 0):.3f}")

    # Summary
    print("\n--- Run Summary ---")
    avg_preferred = sum(r["preferred_count"] for r in all_results) / len(all_results)
    print(f"  Average preferred brand items: {avg_preferred:.1f} / 50")


def test_brand_boost_comparison():
    """Compare results with and without brand preferences."""
    print_separator("TEST 5: With vs Without Brand Preferences")

    module = CandidateSelectionModule()

    # Without brand preferences
    print("\n--- Without Brand Preferences ---")
    no_brand_profile = OnboardingProfile(
        user_id="test_no_brand",
        categories=["tops", "dresses"],
        style_directions=["minimal"]
    )

    no_brand_state = UserState(
        user_id="test_no_brand",
        state_type=UserStateType.TINDER_COMPLETE,
        onboarding_profile=no_brand_profile
    )

    no_brand_candidates = module.get_candidates(no_brand_state, gender="female")

    # With brand preferences
    print("\n--- With Brand Preferences ---")
    with_brand_profile = OnboardingProfile(
        user_id="test_with_brand",
        categories=["tops", "dresses"],
        preferred_brands=["Zara", "H&M", "Mango", "ASOS"],
        brand_openness="stick_to_favorites",
        style_directions=["minimal"]
    )

    with_brand_state = UserState(
        user_id="test_with_brand",
        state_type=UserStateType.TINDER_COMPLETE,
        onboarding_profile=with_brand_profile
    )

    with_brand_candidates = module.get_candidates(with_brand_state, gender="female")

    preferred_brands_lower = ["zara", "h&m", "mango", "asos"]

    # Compare top 20
    print("\n--- Comparison: Top 20 by Final Score ---")
    print(f"{'Rank':<5} {'Without Brands':<35} {'With Brands':<35}")
    print("-" * 75)

    no_brand_sorted = sorted(no_brand_candidates, key=lambda x: x.final_score, reverse=True)[:20]
    with_brand_sorted = sorted(with_brand_candidates, key=lambda x: x.final_score, reverse=True)[:20]

    no_brand_preferred_count = 0
    with_brand_preferred_count = 0

    for i in range(20):
        nb = no_brand_sorted[i] if i < len(no_brand_sorted) else None
        wb = with_brand_sorted[i] if i < len(with_brand_sorted) else None

        nb_str = ""
        wb_str = ""

        if nb:
            is_pref = "★" if (nb.brand or "").lower() in preferred_brands_lower else " "
            nb_str = f"{is_pref}{nb.brand or 'N/A':<14} ({nb.final_score:.3f})"
            if is_pref == "★":
                no_brand_preferred_count += 1

        if wb:
            is_pref = "★" if (wb.brand or "").lower() in preferred_brands_lower else " "
            wb_str = f"{is_pref}{wb.brand or 'N/A':<14} ({wb.final_score:.3f})"
            if is_pref == "★":
                with_brand_preferred_count += 1

        print(f"{i+1:<5} {nb_str:<35} {wb_str:<35}")

    print(f"\nPreferred brands in top 20:")
    print(f"  Without preferences: {no_brand_preferred_count}")
    print(f"  With preferences: {with_brand_preferred_count}")
    print(f"  Improvement: +{with_brand_preferred_count - no_brand_preferred_count}")


def run_all_tests():
    """Run all tests."""
    print_separator("PREFERENCE WEIGHTING AND BRAND PRIORITIZATION TESTS")
    print("""
This test suite validates:
1. Brand boost effectiveness at different openness levels
2. Cold start mitigation with onboarding preferences
3. Type preference filtering
4. Full pipeline integration
5. Comparison with/without brand preferences
""")

    try:
        # Test 1: Brand boost levels
        test_brand_boost_levels()

        # Test 2: Cold start
        test_cold_start_with_onboarding()

        # Test 3: Type filtering
        test_type_filtering()

        # Test 4: Full pipeline
        test_full_pipeline()

        # Test 5: Comparison
        test_brand_boost_comparison()

        print_separator("ALL TESTS COMPLETED")

    except Exception as e:
        print(f"\nError during tests: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
