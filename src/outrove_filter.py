"""
Outrove Candidate Generation Module

This module implements STRICT FILTERING for candidate generation based on
user onboarding preferences. Ranking is handled separately by SASRec.

Pipeline:
1. Filter by category (selectedCoreTypes)
2. Exclude colors (colorsToAvoid)
3. Exclude materials (materialsToAvoid)
4. Filter by price range (per-category)

Uses the pre-computed tops_enriched.pkl file.
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict


# ============================================================================
# DATA CLASSES FOR USER PREFERENCES
# ============================================================================

@dataclass
class GlobalPreferences:
    """Global preferences from Outrove onboarding Step 1."""
    selected_core_types: List[str] = field(default_factory=list)  # e.g., ["t-shirts", "hoodies"]
    typical_size: List[str] = field(default_factory=list)  # e.g., ["L", "XL"]
    colors_to_avoid: List[str] = field(default_factory=list)  # e.g., ["yellow", "mustard"]
    materials_to_avoid: List[str] = field(default_factory=list)  # e.g., ["polyester"]


@dataclass
class TShirtsPreferences:
    """T-Shirts module preferences."""
    size: List[str] = field(default_factory=list)
    fit: Optional[str] = None
    sleeve_length: Optional[str] = None
    length_preference: Optional[str] = None
    necklines: List[str] = field(default_factory=list)
    style_variants: List[str] = field(default_factory=list)
    graphics_tolerance: Optional[str] = None
    price_range: Tuple[float, float] = (10, 50)


@dataclass
class PolosPreferences:
    """Polos module preferences."""
    size: List[str] = field(default_factory=list)
    fit: Optional[str] = None
    style_variants: List[str] = field(default_factory=list)
    pattern_logo_tolerance: Optional[str] = None
    pocket_preference: Optional[str] = None
    price_range: Tuple[float, float] = (30, 100)


@dataclass
class SweaterPreferences:
    """Sweaters module preferences."""
    size: List[str] = field(default_factory=list)
    fit: Optional[str] = None
    necklines: List[str] = field(default_factory=list)
    materials: List[str] = field(default_factory=list)
    weight: Optional[str] = None
    price_range: Tuple[float, float] = (50, 200)


@dataclass
class HoodiesPreferences:
    """Hoodies module preferences."""
    size: List[str] = field(default_factory=list)
    fit: Optional[str] = None
    style_preference: List[str] = field(default_factory=list)
    branding_tolerance: Optional[str] = None
    price_range: Tuple[float, float] = (40, 120)


@dataclass
class ShirtsPreferences:
    """Casual Button-Downs module preferences."""
    size: List[str] = field(default_factory=list)
    fit: Optional[str] = None
    tuck_preference: Optional[str] = None
    fabrics: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    price_range: Tuple[float, float] = (40, 150)


@dataclass
class OutroveUserProfile:
    """Complete user profile from Outrove onboarding."""
    user_id: str
    global_prefs: GlobalPreferences = field(default_factory=GlobalPreferences)
    tshirts: Optional[TShirtsPreferences] = None
    polos: Optional[PolosPreferences] = None
    sweaters: Optional[SweaterPreferences] = None
    hoodies: Optional[HoodiesPreferences] = None
    shirts: Optional[ShirtsPreferences] = None

    @classmethod
    def from_dict(cls, data: Dict) -> 'OutroveUserProfile':
        """Parse user profile from Outrove JSON format."""
        profile = cls(user_id=data.get('user_id', 'anonymous'))

        # Global preferences
        profile.global_prefs = GlobalPreferences(
            selected_core_types=data.get('selectedCoreTypes', []),
            typical_size=data.get('typicalSize', []),
            colors_to_avoid=data.get('colorsToAvoid', []),
            materials_to_avoid=data.get('materialsToAvoid', []),
        )

        # Helper to safely get from dict or None
        def safe_get(d, key, default=None):
            if d is None:
                return default
            return d.get(key, default) if isinstance(d, dict) else default

        t = data.get('tshirts')
        if t is not None and isinstance(t, dict):
            profile.tshirts = TShirtsPreferences(
                size=safe_get(t, 'size', profile.global_prefs.typical_size),
                fit=safe_get(t, 'fit'),
                sleeve_length=safe_get(t, 'sleeveLength'),
                necklines=safe_get(t, 'necklines', []),
                style_variants=safe_get(t, 'styleVariants', []),
                graphics_tolerance=safe_get(t, 'graphicsTolerance'),
                price_range=tuple(safe_get(t, 'priceRange', [10, 50])),
            )

        p = data.get('polos')
        if p is not None and isinstance(p, dict):
            profile.polos = PolosPreferences(
                size=safe_get(p, 'size', profile.global_prefs.typical_size),
                fit=safe_get(p, 'fit'),
                style_variants=safe_get(p, 'styleVariants', []),
                pattern_logo_tolerance=safe_get(p, 'patternLogoTolerance'),
                pocket_preference=safe_get(p, 'pocketPreference'),
                price_range=tuple(safe_get(p, 'priceRange', [30, 100])),
            )

        s = data.get('sweaters')
        if s is not None and isinstance(s, dict):
            profile.sweaters = SweaterPreferences(
                size=safe_get(s, 'size', profile.global_prefs.typical_size),
                fit=safe_get(s, 'fit'),
                necklines=safe_get(s, 'necklines', []),
                materials=safe_get(s, 'materials', []),
                weight=safe_get(s, 'weight'),
                price_range=tuple(safe_get(s, 'priceRange', [50, 200])),
            )

        h = data.get('hoodies')
        if h is not None and isinstance(h, dict):
            profile.hoodies = HoodiesPreferences(
                size=safe_get(h, 'size', profile.global_prefs.typical_size),
                fit=safe_get(h, 'fit'),
                style_preference=safe_get(h, 'stylePreference', []),
                branding_tolerance=safe_get(h, 'brandingTolerance'),
                price_range=tuple(safe_get(h, 'priceRange', [40, 120])),
            )

        sh = data.get('shirts')
        if sh is not None and isinstance(sh, dict):
            profile.shirts = ShirtsPreferences(
                size=safe_get(sh, 'size', profile.global_prefs.typical_size),
                fit=safe_get(sh, 'fit'),
                tuck_preference=safe_get(sh, 'tuckPreference'),
                fabrics=safe_get(sh, 'fabrics', []),
                patterns=safe_get(sh, 'patterns', []),
                price_range=tuple(safe_get(sh, 'priceRange', [40, 150])),
            )

        return profile


# ============================================================================
# OUTROVE CANDIDATE FILTER (Strict Filtering Only)
# ============================================================================

class OutroveCandidateFilter:
    """
    Strict candidate filtering for Outrove recommendations.

    This module handles ONLY candidate generation through strict filtering.
    Ranking is done by SASRec separately.

    Filters applied:
    1. Category (selectedCoreTypes)
    2. Colors to avoid
    3. Materials to avoid
    4. Price range (per category)
    """

    def __init__(self, tops_data_path: str):
        """
        Initialize filter with pre-computed tops data.

        Args:
            tops_data_path: Path to tops_enriched.pkl
        """
        print(f"Loading tops data from {tops_data_path}...")
        with open(tops_data_path, 'rb') as f:
            self.data = pickle.load(f)

        self.items = self.data['items']
        self.embeddings = self.data.get('embeddings')
        self.item_ids = self.data.get('item_ids', [])
        self.query_embeddings = self.data.get('query_embeddings', {})
        self.stats = self.data.get('stats', {})

        # Build indices for fast filtering
        self._build_indices()

        print(f"Loaded {len(self.items)} tops items")
        print(f"  Categories: {self.stats.get('category_counts', {})}")

    def _build_indices(self):
        """Build inverted indices for fast filtering."""
        # Category index
        self.by_category: Dict[str, Set[str]] = defaultdict(set)
        for item_id, item in self.items.items():
            self.by_category[item.get('outrove_type', '')].add(item_id)

        # Color index (items containing each color)
        self.by_color: Dict[str, Set[str]] = defaultdict(set)
        for item_id, item in self.items.items():
            for color in item.get('colors', []):
                self.by_color[color].add(item_id)

        # Material index
        self.by_material: Dict[str, Set[str]] = defaultdict(set)
        for item_id, item in self.items.items():
            for material in item.get('materials', []):
                self.by_material[material].add(item_id)

        # Build item_id -> embedding index mapping
        self.item_to_emb_idx = {item_id: idx for idx, item_id in enumerate(self.item_ids)}

        # All item IDs as a set for fast lookup
        self.all_item_ids = set(self.items.keys())

    def get_item(self, item_id: str) -> Optional[Dict]:
        """Get enriched item data by ID."""
        return self.items.get(item_id)

    def get_all_items(self) -> Dict[str, Dict]:
        """Get all items."""
        return self.items

    def get_all_item_ids(self) -> Set[str]:
        """Get all item IDs."""
        return self.all_item_ids

    # ========================================================================
    # STRICT FILTERING
    # ========================================================================

    def filter_by_category(self, categories: List[str]) -> Set[str]:
        """Filter items by Outrove category types."""
        if not categories:
            return set(self.items.keys())

        result = set()
        for cat in categories:
            result.update(self.by_category.get(cat, set()))
        return result

    def filter_exclude_colors(self, colors_to_avoid: List[str], candidates: Set[str]) -> Set[str]:
        """Remove items containing avoided colors."""
        if not colors_to_avoid:
            return candidates

        items_with_avoided_colors = set()
        for color in colors_to_avoid:
            items_with_avoided_colors.update(self.by_color.get(color, set()))

        return candidates - items_with_avoided_colors

    def filter_exclude_materials(self, materials_to_avoid: List[str], candidates: Set[str]) -> Set[str]:
        """Remove items containing avoided materials."""
        if not materials_to_avoid:
            return candidates

        items_with_avoided_materials = set()
        for material in materials_to_avoid:
            items_with_avoided_materials.update(self.by_material.get(material, set()))

        return candidates - items_with_avoided_materials

    def filter_by_price(self, price_range: Tuple[float, float], candidates: Set[str]) -> Set[str]:
        """Filter items by price range. Items without price are kept."""
        if not price_range:
            return candidates

        min_price, max_price = price_range
        result = set()
        for item_id in candidates:
            item = self.items.get(item_id, {})
            price = item.get('price')
            if price is None:
                # Keep items without price (benefit of doubt)
                result.add(item_id)
            elif min_price <= price <= max_price:
                result.add(item_id)

        return result

    def _get_category_prefs(self, profile: OutroveUserProfile, category: str):
        """Get preferences for a specific category."""
        mapping = {
            't-shirts': profile.tshirts,
            'polos': profile.polos,
            'sweaters': profile.sweaters,
            'hoodies': profile.hoodies,
            'shirts': profile.shirts,
        }
        return mapping.get(category)

    def get_candidates(self, profile: OutroveUserProfile) -> Set[str]:
        """
        Get filtered candidate item IDs based on user profile.

        Applies strict filters:
        1. Category filter
        2. Color exclusion
        3. Material exclusion
        4. Price range (per category)

        Returns:
            Set of item IDs that pass all filters
        """
        all_candidates = set()

        for category in profile.global_prefs.selected_core_types:
            # Start with all items in this category
            candidates = self.filter_by_category([category])

            # Apply global exclusions
            candidates = self.filter_exclude_colors(
                profile.global_prefs.colors_to_avoid,
                candidates
            )
            candidates = self.filter_exclude_materials(
                profile.global_prefs.materials_to_avoid,
                candidates
            )

            # Apply category-specific price filter
            prefs = self._get_category_prefs(profile, category)
            if prefs and hasattr(prefs, 'price_range'):
                candidates = self.filter_by_price(prefs.price_range, candidates)

            all_candidates.update(candidates)

        return all_candidates

    def get_candidates_with_scores(self, profile: OutroveUserProfile) -> List[Tuple[str, float]]:
        """
        Get filtered candidate item IDs with match scores based on user profile.

        Uses OR logic for positive preferences (necklines, materials, fits, etc.)
        and ranks items by how many preferences they match.

        Hard filters (AND logic - must pass all):
        - Category (selectedCoreTypes)
        - Colors to avoid
        - Materials to avoid
        - Price range

        Soft filters (OR logic - match any, boost score):
        - Preferred necklines (from category prefs)
        - Preferred materials (from sweater prefs)
        - Preferred fit

        Returns:
            List of (item_id, score) tuples sorted by score descending
        """
        # First get candidates that pass hard filters
        all_candidates = self.get_candidates(profile)

        # Score each candidate based on positive preference matches
        scored_candidates = []

        for item_id in all_candidates:
            item = self.items.get(item_id, {})
            score = 1.0  # Base score
            match_count = 0

            item_category = item.get('outrove_type', '')
            prefs = self._get_category_prefs(profile, item_category)

            if prefs:
                # Check necklines (t-shirts, sweaters)
                if hasattr(prefs, 'necklines') and prefs.necklines:
                    item_neckline = (item.get('neckline') or '').lower()
                    for pref_neckline in prefs.necklines:
                        if item_neckline and (pref_neckline.lower() in item_neckline or item_neckline in pref_neckline.lower()):
                            match_count += 1
                            break

                # Check fit preference
                if hasattr(prefs, 'fit') and prefs.fit:
                    item_fit = (item.get('fit') or '').lower()
                    visual_fit = (item.get('visual_fit') or '').lower()
                    pref_fit = prefs.fit.lower()
                    if (item_fit and pref_fit in item_fit) or (visual_fit and pref_fit in visual_fit):
                        match_count += 1

                # Check materials (sweaters)
                if hasattr(prefs, 'materials') and prefs.materials:
                    item_materials = [m.lower() for m in (item.get('materials') or [])]
                    for pref_mat in prefs.materials:
                        if any(pref_mat.lower() in m for m in item_materials):
                            match_count += 1
                            break

                # Check style variants (t-shirts, hoodies)
                if hasattr(prefs, 'style_variants') and prefs.style_variants:
                    item_pattern = (item.get('pattern') or '').lower()
                    visual_pattern = (item.get('visual_pattern') or '').lower()
                    for style in prefs.style_variants:
                        style_lower = style.lower()
                        # Map style variants to patterns
                        if style_lower == 'plain' and ('solid' in item_pattern or 'plain' in item_pattern):
                            match_count += 1
                            break
                        elif style_lower == 'graphic-tees' and ('graphic' in item_pattern or 'print' in visual_pattern):
                            match_count += 1
                            break
                        elif style_lower in item_pattern or style_lower in visual_pattern:
                            match_count += 1
                            break

                # Check fabrics (shirts)
                if hasattr(prefs, 'fabrics') and prefs.fabrics:
                    item_materials = [m.lower() for m in (item.get('materials') or [])]
                    for fabric in prefs.fabrics:
                        if any(fabric.lower() in m for m in item_materials):
                            match_count += 1
                            break

                # Check patterns (shirts)
                if hasattr(prefs, 'patterns') and prefs.patterns:
                    item_pattern = (item.get('pattern') or '').lower()
                    visual_pattern = (item.get('visual_pattern') or '').lower()
                    for pattern in prefs.patterns:
                        if pattern.lower() in item_pattern or pattern.lower() in visual_pattern:
                            match_count += 1
                            break

            # Calculate final score (base + 0.1 per match)
            score = 1.0 + (match_count * 0.1)
            scored_candidates.append((item_id, score))

        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates

    def get_candidates_by_category(self, profile: OutroveUserProfile) -> Dict[str, Set[str]]:
        """
        Get filtered candidates grouped by category.

        Returns:
            Dict mapping category -> set of item IDs
        """
        results = {}

        for category in profile.global_prefs.selected_core_types:
            candidates = self.filter_by_category([category])
            candidates = self.filter_exclude_colors(
                profile.global_prefs.colors_to_avoid,
                candidates
            )
            candidates = self.filter_exclude_materials(
                profile.global_prefs.materials_to_avoid,
                candidates
            )

            prefs = self._get_category_prefs(profile, category)
            if prefs and hasattr(prefs, 'price_range'):
                candidates = self.filter_by_price(prefs.price_range, candidates)

            results[category] = candidates

        return results

    def get_filter_stats(self, profile: OutroveUserProfile) -> Dict[str, Any]:
        """
        Get statistics about filtering.

        Returns:
            Dict with before/after counts and filter rates
        """
        candidates_by_cat = self.get_candidates_by_category(profile)

        total_before = sum(
            len(self.by_category.get(cat, set()))
            for cat in profile.global_prefs.selected_core_types
        )
        total_after = sum(len(items) for items in candidates_by_cat.values())

        return {
            "selected_categories": profile.global_prefs.selected_core_types,
            "filters_applied": {
                "colors_to_avoid": profile.global_prefs.colors_to_avoid,
                "materials_to_avoid": profile.global_prefs.materials_to_avoid,
            },
            "items_before_filtering": total_before,
            "items_after_filtering": total_after,
            "items_filtered_out": total_before - total_after,
            "filter_rate_percent": round((total_before - total_after) / total_before * 100, 1) if total_before > 0 else 0,
            "by_category": {cat: len(items) for cat, items in candidates_by_cat.items()},
        }

    def get_stats(self) -> Dict:
        """Get statistics about the loaded data."""
        return {
            "total_items": len(self.items),
            "items_per_category": {cat: len(ids) for cat, ids in self.by_category.items()},
            "has_embeddings": self.embeddings is not None,
            "num_visual_queries": len(self.query_embeddings),
            **self.stats,
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_filter(data_dir: str = None) -> OutroveCandidateFilter:
    """
    Create an OutroveCandidateFilter with default paths.

    Args:
        data_dir: Project root directory. Auto-detected if not provided.
    """
    if data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.dirname(script_dir)

    tops_path = os.path.join(data_dir, "data/amazon_fashion/processed/tops_enriched.pkl")
    return OutroveCandidateFilter(tops_path)


def parse_user_profile(data: Dict) -> OutroveUserProfile:
    """Parse user profile from Outrove JSON."""
    return OutroveUserProfile.from_dict(data)


# ============================================================================
# DEMO / TEST
# ============================================================================

if __name__ == "__main__":
    # Create filter
    filter = create_filter()
    print("\nFilter stats:", filter.get_stats())

    # Example user profile
    example_profile = {
        "user_id": "demo_user",
        "selectedCoreTypes": ["t-shirts", "hoodies"],
        "typicalSize": ["L", "XL"],
        "colorsToAvoid": ["yellow", "mustard", "pink"],
        "materialsToAvoid": ["polyester"],
        "tshirts": {
            "size": ["L", "XL"],
            "fit": "relaxed",
            "sleeveLength": "short",
            "necklines": ["crew"],
            "styleVariants": ["plain", "graphic-tees"],
            "graphicsTolerance": "graphics-ok",
            "priceRange": [20, 60]
        },
        "hoodies": {
            "size": ["L"],
            "fit": "oversized",
            "stylePreference": ["pullover"],
            "brandingTolerance": "small-logos",
            "priceRange": [50, 100]
        }
    }

    profile = parse_user_profile(example_profile)
    print(f"\nUser profile: {profile.user_id}")
    print(f"  Core types: {profile.global_prefs.selected_core_types}")
    print(f"  Colors to avoid: {profile.global_prefs.colors_to_avoid}")

    # Get filter stats
    stats = filter.get_filter_stats(profile)
    print(f"\nFilter Stats:")
    print(f"  Before: {stats['items_before_filtering']}")
    print(f"  After: {stats['items_after_filtering']}")
    print(f"  Filter rate: {stats['filter_rate_percent']}%")
    print(f"  By category: {stats['by_category']}")

    # Get candidates
    candidates = filter.get_candidates(profile)
    print(f"\nTotal candidates: {len(candidates)}")

    # Show a few examples
    print("\nSample candidates:")
    for item_id in list(candidates)[:5]:
        item = filter.get_item(item_id)
        print(f"  {item.get('title', 'N/A')[:60]}...")
        print(f"     Category: {item.get('outrove_type')}, Price: ${item.get('price', 'N/A')}")
