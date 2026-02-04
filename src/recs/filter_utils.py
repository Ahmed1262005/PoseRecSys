"""
Shared Filter Utilities Module

Centralizes filtering logic used by both feed pipeline and search engine:
- Image hash extraction and deduplication
- Diversity constraints (max per category)
- Soft preference scoring weights

This module eliminates duplication between:
- pipeline.py (feed path)
- women_search_engine.py (search path)
"""

import re
from typing import List, Dict, Optional, Set, Any, Union, TypeVar, Protocol
from dataclasses import dataclass, field
from collections import defaultdict


# =============================================================================
# Type Definitions
# =============================================================================

class HasImageUrl(Protocol):
    """Protocol for objects with image_url attribute."""
    image_url: Optional[str]


class HasNameBrand(Protocol):
    """Protocol for objects with name and brand attributes."""
    name: Optional[str]
    brand: Optional[str]


class HasCategory(Protocol):
    """Protocol for objects with category attributes."""
    broad_category: Optional[str]
    category: Optional[str]


T = TypeVar('T')


# =============================================================================
# Image Hash Extraction
# =============================================================================

# Regex pattern for extracting image hash from URLs
# Matches URLs like: .../original_0_85a218f8.jpg -> hash is 85a218f8
IMAGE_HASH_PATTERN = re.compile(r'original_\d+_([a-f0-9]+)\.')


def extract_image_hash(url: Optional[str]) -> Optional[str]:
    """
    Extract image hash from URL.

    Many fashion retailers (e.g., Boohoo/Nasty Gal) sell identical items under
    different brand names. This extracts the image hash to identify duplicates.

    Args:
        url: Image URL like .../original_0_85a218f8.jpg

    Returns:
        Hash string (e.g., "85a218f8") or None if not found
    """
    if not url:
        return None
    match = IMAGE_HASH_PATTERN.search(url)
    return match.group(1) if match else None


# =============================================================================
# Deduplication
# =============================================================================

def deduplicate_items(
    items: List[T],
    get_image_url: callable,
    get_name: callable,
    get_brand: callable,
    seen_hashes: Optional[Set[str]] = None,
    seen_name_brand: Optional[Set[tuple]] = None,
    limit: Optional[int] = None
) -> List[T]:
    """
    Generic deduplication for any item type (Candidate or Dict).

    Handles two types of duplicates:
    1. Same image across different brands (cross-brand duplicates)
    2. Same name+brand with different IDs (same product scraped multiple times)

    Args:
        items: List of items to deduplicate
        get_image_url: Function to extract image URL from item
        get_name: Function to extract name from item
        get_brand: Function to extract brand from item
        seen_hashes: Optional existing set of seen image hashes (mutated)
        seen_name_brand: Optional existing set of seen (name, brand) tuples (mutated)
        limit: Optional limit on number of results

    Returns:
        Deduplicated list of items
    """
    if seen_hashes is None:
        seen_hashes = set()
    if seen_name_brand is None:
        seen_name_brand = set()

    deduped = []

    for item in items:
        # Primary dedup: image hash (catches cross-brand duplicates)
        img_url = get_image_url(item)
        img_hash = extract_image_hash(img_url)
        if img_hash and img_hash in seen_hashes:
            continue  # Skip - same image already shown

        # Secondary dedup: (name, brand) for products without matching image hash
        name = (get_name(item) or '').lower().strip()
        brand = (get_brand(item) or '').lower().strip()
        name_brand_key = (name, brand) if name and brand else None
        if name_brand_key and name_brand_key in seen_name_brand:
            continue  # Skip - same name+brand already shown

        deduped.append(item)
        if img_hash:
            seen_hashes.add(img_hash)
        if name_brand_key:
            seen_name_brand.add(name_brand_key)

        # Stop if we have enough results
        if limit and len(deduped) >= limit:
            break

    return deduped


def deduplicate_candidates(
    candidates: List[Any],
    seen_hashes: Optional[Set[str]] = None,
    seen_name_brand: Optional[Set[tuple]] = None
) -> List[Any]:
    """
    Deduplicate Candidate objects by image hash and name+brand.

    Args:
        candidates: List of Candidate objects
        seen_hashes: Optional existing set of seen image hashes
        seen_name_brand: Optional existing set of seen (name, brand) tuples

    Returns:
        Deduplicated list of Candidate objects
    """
    return deduplicate_items(
        items=candidates,
        get_image_url=lambda c: c.image_url,
        get_name=lambda c: c.name,
        get_brand=lambda c: c.brand,
        seen_hashes=seen_hashes,
        seen_name_brand=seen_name_brand
    )


def deduplicate_dicts(
    results: List[Dict],
    seen_hashes: Optional[Set[str]] = None,
    seen_name_brand: Optional[Set[tuple]] = None,
    limit: Optional[int] = None
) -> List[Dict]:
    """
    Deduplicate Dict results by image hash and name+brand.

    Args:
        results: List of result dicts
        seen_hashes: Optional existing set of seen image hashes
        seen_name_brand: Optional existing set of seen (name, brand) tuples
        limit: Optional limit on number of results

    Returns:
        Deduplicated list of result dicts
    """
    return deduplicate_items(
        items=results,
        get_image_url=lambda d: d.get('image_url') or d.get('primary_image_url'),
        get_name=lambda d: d.get('name'),
        get_brand=lambda d: d.get('brand'),
        seen_hashes=seen_hashes,
        seen_name_brand=seen_name_brand,
        limit=limit
    )


# =============================================================================
# Diversity Constraints
# =============================================================================

@dataclass
class DiversityConfig:
    """
    Configuration for category diversity constraints.

    Controls maximum items per category to ensure varied results.
    """
    # Default max items per category
    default_limit: int = 8

    # Per-category overrides (broad_category -> max count)
    category_limits: Dict[str, int] = field(default_factory=dict)

    # Dynamic adjustment based on user's selected categories
    single_category_limit: int = 50    # When user selected 1 category
    two_category_limit: int = 25       # When user selected 2 categories
    three_category_limit: int = 16     # When user selected 3 categories

    # Limit for warm users with taste vector (be more lenient)
    warm_user_limit: int = 50


# Default diversity config (shared between feed and search)
DEFAULT_DIVERSITY_CONFIG = DiversityConfig()


def get_diversity_limit(
    user_categories: Optional[List[str]] = None,
    has_taste_vector: bool = False,
    config: Optional[DiversityConfig] = None
) -> int:
    """
    Calculate dynamic diversity limit based on user state.

    Args:
        user_categories: User's selected broad categories
        has_taste_vector: Whether user has completed style discovery
        config: Diversity configuration

    Returns:
        Max items per category
    """
    config = config or DEFAULT_DIVERSITY_CONFIG

    # Warm users with taste vector get more lenient limits
    # Similarity search already biases toward liked categories
    if has_taste_vector:
        return config.warm_user_limit

    # Dynamic limit based on number of selected categories
    num_categories = len(user_categories) if user_categories else 0

    if num_categories <= 1:
        return config.single_category_limit
    elif num_categories == 2:
        return config.two_category_limit
    elif num_categories == 3:
        return config.three_category_limit
    else:
        return config.default_limit


def apply_diversity_candidates(
    candidates: List[Any],
    max_per_category: int,
    get_category: Optional[callable] = None
) -> List[Any]:
    """
    Apply diversity constraints to Candidate objects.

    Args:
        candidates: List of Candidate objects
        max_per_category: Max items per category
        get_category: Optional function to get category (defaults to broad_category or category)

    Returns:
        Diversity-constrained list
    """
    if get_category is None:
        get_category = lambda c: c.broad_category or c.category or "unknown"

    result = []
    category_counts = defaultdict(int)

    for candidate in candidates:
        cat = get_category(candidate)

        if category_counts[cat] < max_per_category:
            result.append(candidate)
            category_counts[cat] += 1

    return result


def apply_diversity_dicts(
    results: List[Dict],
    max_per_category: int = 15
) -> List[Dict]:
    """
    Apply diversity constraints to Dict results.

    Args:
        results: List of result dicts
        max_per_category: Max items per category

    Returns:
        Diversity-constrained list
    """
    category_counts = defaultdict(int)
    diverse_results = []

    for item in results:
        cat = item.get('broad_category') or item.get('category') or 'unknown'

        if category_counts[cat] < max_per_category:
            diverse_results.append(item)
            category_counts[cat] += 1

    return diverse_results


# =============================================================================
# Soft Scoring Weights
# =============================================================================

@dataclass
class SoftScoringWeights:
    """
    Centralized soft preference scoring weights.

    IMPORTANT: Boosts are CAPPED to prevent overriding semantic relevance.
    - MAX_TOTAL_BOOST: Positive boosts never exceed this
    - SEMANTIC_FLOOR: Items below this semantic score get no boosts

    BOOST weights (positive preferences):
    - Fit match, sleeve match, length match, rise match
    - Preferred brand, type match

    DEMOTE weights (things to avoid):
    - Color to avoid, style to avoid, material to avoid, brand to avoid
    """
    # Caps to prevent boosts from overriding semantic relevance
    max_total_boost: float = 0.15
    semantic_floor: float = 0.25  # Text-to-image CLIP scores typically 0.28-0.40

    # Boost weights (positive preferences) - reduced to prevent override
    fit_boost: float = 0.03
    sleeve_boost: float = 0.03
    length_boost: float = 0.03
    rise_boost: float = 0.02
    brand_boost: float = 0.05
    type_boost: float = 0.03

    # Demote weights (things to avoid) - negative values
    color_demote: float = -0.15
    style_demote: float = -0.12
    material_demote: float = -0.10
    brand_demote: float = -0.15


# Default weights (shared between feed and search)
DEFAULT_SOFT_WEIGHTS = SoftScoringWeights()


def compute_soft_score_boost(
    item: Dict[str, Any],
    soft_prefs: Dict[str, Any],
    type_prefs: Dict[str, Any],
    hard_filters: Optional[Dict[str, Any]] = None,
    weights: Optional[SoftScoringWeights] = None
) -> tuple[float, List[str], List[str]]:
    """
    Compute soft preference score boost for an item.

    Args:
        item: Item dict with attributes (fit, sleeve, brand, colors, materials, etc.)
        soft_prefs: User's soft preferences (preferred_fits, preferred_sleeves, etc.)
        type_prefs: User's type preferences (top_types, bottom_types, etc.)
        hard_filters: User's hard filters for demotes (exclude_colors, exclude_materials, etc.)
        weights: Scoring weights configuration

    Returns:
        Tuple of (total_boost, match_reasons, demote_reasons)
    """
    weights = weights or DEFAULT_SOFT_WEIGHTS
    hard_filters = hard_filters or {}

    positive_boost = 0.0
    negative_boost = 0.0
    match_reasons = []
    demote_reasons = []

    # Get base semantic score
    base_semantic = item.get('base_similarity', item.get('similarity', 0))

    # Extract item attributes
    item_brand = (item.get('brand') or '').lower()
    item_colors = [c.lower() for c in (item.get('colors') or [])]
    item_materials = [m.lower() for m in (item.get('materials') or [])]

    # === Build preference sets ===
    preferred_fits = {f.lower() for f in soft_prefs.get('preferred_fits', [])}
    preferred_sleeves = {s.lower() for s in soft_prefs.get('preferred_sleeves', [])}
    preferred_lengths = {l.lower() for l in soft_prefs.get('preferred_lengths', [])}
    preferred_lengths_dresses = {l.lower() for l in soft_prefs.get('preferred_lengths_dresses', [])}
    preferred_rises = {r.lower() for r in soft_prefs.get('preferred_rises', [])}
    preferred_brands = {b.lower() for b in soft_prefs.get('preferred_brands', [])}

    # Type preferences by category
    all_type_prefs = set()
    for types in type_prefs.values():
        all_type_prefs.update(t.lower() for t in (types or []))

    # === Build demote sets ===
    colors_to_avoid = {c.lower() for c in hard_filters.get('exclude_colors', []) or []}
    styles_to_avoid = {s.lower() for s in hard_filters.get('exclude_styles', []) or []}
    materials_to_avoid = {m.lower() for m in hard_filters.get('exclude_materials', []) or []}
    brands_to_avoid = {b.lower() for b in hard_filters.get('exclude_brands', []) or []}

    # === BOOSTS (only apply if semantic score meets floor) ===
    if base_semantic >= weights.semantic_floor:
        # Fit match
        item_fit = (item.get('fit') or '').lower()
        if item_fit and item_fit in preferred_fits:
            positive_boost += weights.fit_boost
            match_reasons.append('fit')

        # Sleeve match
        item_sleeve = (item.get('sleeve') or '').lower()
        if item_sleeve and item_sleeve in preferred_sleeves:
            positive_boost += weights.sleeve_boost
            match_reasons.append('sleeve')

        # Length match (use dress lengths for dresses, regular for others)
        item_length = (item.get('length') or '').lower()
        item_category = (item.get('broad_category') or '').lower()
        if item_length:
            if item_category in ['dresses', 'skirts']:
                if item_length in preferred_lengths_dresses:
                    positive_boost += weights.length_boost
                    match_reasons.append('length')
            else:
                if item_length in preferred_lengths:
                    positive_boost += weights.length_boost
                    match_reasons.append('length')

        # Rise match (for bottoms)
        item_rise = (item.get('rise') or '').lower()
        if item_rise and item_rise in preferred_rises:
            positive_boost += weights.rise_boost
            match_reasons.append('rise')

        # Brand match (boost preferred brands)
        if item_brand and item_brand in preferred_brands:
            positive_boost += weights.brand_boost
            match_reasons.append('brand')

        # Type match
        item_type = (item.get('article_type') or '').lower()
        if item_type and item_type in all_type_prefs:
            positive_boost += weights.type_boost
            match_reasons.append('type')

    # === DEMOTES (always apply, no floor) ===

    # Demote colors to avoid (partial match)
    if colors_to_avoid and item_colors:
        for item_color in item_colors:
            if any(avoided in item_color for avoided in colors_to_avoid):
                negative_boost += weights.color_demote
                demote_reasons.append('color_avoid')
                break  # Only demote once per item

    # Demote materials to avoid (partial match)
    if materials_to_avoid and item_materials:
        for item_material in item_materials:
            if any(avoided in item_material for avoided in materials_to_avoid):
                negative_boost += weights.material_demote
                demote_reasons.append('material_avoid')
                break  # Only demote once per item

    # Demote brands to avoid
    if brands_to_avoid and item_brand:
        if item_brand in brands_to_avoid:
            negative_boost += weights.brand_demote
            demote_reasons.append('brand_avoid')

    # Cap positive boosts to prevent overriding semantic relevance
    capped_positive_boost = min(weights.max_total_boost, positive_boost)
    total_boost = capped_positive_boost + negative_boost  # negative_boost is already negative

    return total_boost, match_reasons, demote_reasons


def apply_soft_scoring(
    results: List[Dict],
    soft_prefs: Dict[str, Any],
    type_prefs: Dict[str, Any],
    hard_filters: Optional[Dict[str, Any]] = None,
    weights: Optional[SoftScoringWeights] = None
) -> List[Dict]:
    """
    Apply soft preference scoring to search results.

    Modifies results in-place and re-sorts by boosted similarity.

    Args:
        results: List of result dicts
        soft_prefs: User's soft preferences
        type_prefs: User's type preferences
        hard_filters: User's hard filters for demotes
        weights: Scoring weights configuration

    Returns:
        Results with updated similarity scores, sorted by boosted score
    """
    if not soft_prefs and not type_prefs and not hard_filters:
        return results

    for item in results:
        total_boost, match_reasons, demote_reasons = compute_soft_score_boost(
            item, soft_prefs, type_prefs, hard_filters, weights
        )

        # Apply boost to similarity score (clamp to 0-1 range)
        original_similarity = item.get('similarity', 0)
        item['similarity'] = max(0.0, min(1.0, original_similarity + total_boost))
        item['preference_boost'] = round(total_boost, 4)
        item['preference_matches'] = match_reasons
        item['preference_demotes'] = demote_reasons

    # Re-sort by boosted similarity
    results.sort(key=lambda x: x.get('similarity', 0), reverse=True)

    return results
