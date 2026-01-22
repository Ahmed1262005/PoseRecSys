"""
Attribute Isolation Style Learning Engine

A structured approach to learning user preferences by testing
ONE attribute at a time while keeping other variables constant.

Key insight: Show 4 items from the SAME visual cluster (similar look)
that differ ONLY on the attribute being tested. This isolates the
variable and gives clear signal on what the user prefers.

Phases:
1. Style Foundation - Discover archetype preference
2. Category Exploration - What type of item (plain vs graphic)
3. Fit & Form - How should it fit
4. Visual Elements - Logo position, pattern preferences
5. Color Palette - Color family preferences
6. Material Feel - Fabric preferences

Each phase has multiple rounds to build confidence.
"""

import numpy as np
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import random

from swipe_engine import SwipeEngine, UserPreferences


# =============================================================================
# ATTRIBUTE DEFINITIONS
# =============================================================================

ATTRIBUTE_TEST_PHASES = [
    {
        'phase': 1,
        'name': 'Style Foundation',
        'icon': '🎨',
        'description': 'Discover your overall style vibe',
        'attributes': ['archetype'],
        'rounds_per_attr': 2,
    },
    {
        'phase': 2,
        'name': 'Category Exploration',
        'icon': '👕',
        'description': 'What type of top speaks to you',
        'attributes': ['category', 'logo_style'],
        'rounds_per_attr': 2,
    },
    {
        'phase': 3,
        'name': 'Fit & Form',
        'icon': '📐',
        'description': 'Find your ideal fit and shape',
        'attributes': ['fit', 'neckline'],
        'rounds_per_attr': 2,
    },
    {
        'phase': 4,
        'name': 'Visual Elements',
        'icon': '✨',
        'description': 'Define your graphic preferences',
        'attributes': ['pattern_density'],
        'rounds_per_attr': 2,
    },
    {
        'phase': 5,
        'name': 'Color Palette',
        'icon': '🎨',
        'description': 'Build your color profile',
        'attributes': ['color_family', 'color_tone'],
        'rounds_per_attr': 2,
    },
    {
        'phase': 6,
        'name': 'Material & Quality',
        'icon': '🧵',
        'description': 'Texture and fabric preferences',
        'attributes': ['material'],
        'rounds_per_attr': 1,
    },
]

# Attribute value definitions
ATTRIBUTE_VALUES = {
    'archetype': ['classic', 'creative_artistic', 'natural_sporty', 'minimalist', 'bold_expressive'],
    'category': ['Plain T-shirts', 'Small logos', 'Graphics T-shirts', 'Athletic'],
    'logo_style': ['none', 'minimal', 'statement', 'all_over'],
    'fit': ['Slim', 'Regular', 'Relaxed', 'Oversized'],
    'neckline': ['crew', 'v_neck', 'henley', 'polo'],
    'pattern_density': ['solid', 'minimal', 'moderate', 'busy'],
    'color_family': ['dark', 'light', 'cool', 'warm', 'earth'],
    'color_tone': ['neutral', 'muted', 'vibrant', 'pastel'],
    'material': ['cotton', 'polyester', 'blend', 'premium'],
}

# Color classification
COLOR_FAMILIES = {
    'dark': {'black', 'navy', 'dark grey', 'dark navy', 'charcoal', 'dark blue'},
    'light': {'white', 'cream', 'beige', 'off-white', 'light grey', 'sand', 'oat milk', 'ivory'},
    'cool': {'blue', 'light blue', 'teal', 'purple', 'lavender', 'mint', 'sky blue'},
    'warm': {'red', 'orange', 'yellow', 'pink', 'coral', 'burnt orange', 'peach', 'salmon'},
    'earth': {'brown', 'olive', 'tan', 'khaki', 'olive green', 'forest green', 'burgundy', 'rust'},
}

COLOR_TONES = {
    'neutral': {'black', 'white', 'grey', 'gray', 'charcoal', 'light grey', 'dark grey'},
    'muted': {'navy', 'olive', 'burgundy', 'forest green', 'rust', 'mauve', 'dusty'},
    'vibrant': {'red', 'blue', 'yellow', 'green', 'orange', 'purple', 'pink', 'teal'},
    'pastel': {'light blue', 'lavender', 'mint', 'peach', 'coral', 'baby blue', 'blush'},
}

# Logo style inference from category
CATEGORY_TO_LOGO_STYLE = {
    'Plain T-shirts': 'none',
    'Small logos': 'minimal',
    'Graphics T-shirts': 'statement',
    'Athletic': 'minimal',
}

# Pattern density inference
CATEGORY_TO_PATTERN = {
    'Plain T-shirts': 'solid',
    'Small logos': 'minimal',
    'Graphics T-shirts': 'moderate',  # Could be busy too
    'Athletic': 'minimal',
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AttributeTestPreferences(UserPreferences):
    """Extended preferences for attribute isolation testing."""

    # Current test state
    current_phase: int = 1
    current_attribute_index: int = 0
    rounds_on_current_attr: int = 0

    # Learned preferences (attribute -> value -> score)
    attribute_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Locked preferences (confirmed from multiple rounds)
    locked_attributes: Dict[str, str] = field(default_factory=dict)

    # Confidence scores for each locked attribute
    attribute_confidence: Dict[str, float] = field(default_factory=dict)

    # Reference embedding for visual similarity (from first liked item)
    reference_embedding: np.ndarray = None
    reference_cluster: int = None

    # Detailed history
    test_history: List[Dict] = field(default_factory=list)
    rounds_completed: int = 0

    # Items shown in current round
    current_four: List[str] = field(default_factory=list)
    current_test_attribute: str = None

    # Phase tracking
    phase_complete: Set[int] = field(default_factory=set)


# =============================================================================
# ENGINE
# =============================================================================

class AttributeTestEngine(SwipeEngine):
    """
    Attribute Isolation Engine

    Strategy: For each attribute being tested:
    1. Find items that are VISUALLY SIMILAR (same cluster or close embeddings)
    2. But DIFFER on the specific attribute being tested
    3. Show 4 items, each with a different value for that attribute
    4. User's choice reveals their preference for that attribute
    5. Lock in preference after 2 confirmations
    6. Move to next attribute

    This isolates variables like a controlled experiment.
    """

    # Total rounds calculation
    TOTAL_PHASES = len(ATTRIBUTE_TEST_PHASES)
    MIN_ROUNDS = 12  # Minimum before allowing early exit
    MAX_ROUNDS = 25  # Hard cap

    # Visual similarity threshold for "same look" grouping
    SIMILARITY_THRESHOLD = 0.75

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Pre-compute attribute values for all items
        self._precompute_item_attributes()

        # Build attribute-based indexes for fast lookup
        self._build_attribute_indexes()

    def _precompute_item_attributes(self):
        """Pre-compute all attribute values for each item."""
        self.item_attributes = {}

        for item_id in self.item_ids:
            meta = self.metadata.get(item_id, {})
            emb = self.embeddings_data.get(item_id, {}).get('embedding')
            cluster = self.item_to_cluster.get(item_id)

            # Get archetype from cluster profile if not in metadata
            archetype = meta.get('dominant_archetype', meta.get('archetype', ''))
            if not archetype or archetype == 'unknown':
                cluster_profile = self._get_cluster_profile(cluster) if cluster is not None else {}
                archetype = cluster_profile.get('dominant_archetype', 'classic')

            attrs = {
                'archetype': archetype if archetype else 'classic',
                'category': meta.get('category', ''),
                'logo_style': self._infer_logo_style(meta),
                'fit': meta.get('fit', 'Regular'),
                'neckline': self._infer_neckline(meta),
                'pattern_density': self._infer_pattern_density(meta),
                'color_family': self._classify_color_family(meta.get('color', '')),
                'color_tone': self._classify_color_tone(meta.get('color', '')),
                'material': self._classify_material(meta.get('fabric', '')),
                'cluster': cluster,
            }

            self.item_attributes[item_id] = attrs

    def _build_attribute_indexes(self):
        """Build indexes for fast attribute-based lookup."""
        self.attr_to_items = defaultdict(lambda: defaultdict(list))

        for item_id, attrs in self.item_attributes.items():
            for attr_name, attr_value in attrs.items():
                if attr_value and attr_value != 'unknown':
                    self.attr_to_items[attr_name][attr_value].append(item_id)

    def _infer_logo_style(self, meta: Dict) -> str:
        """Infer logo style from category and other metadata."""
        category = meta.get('category', '')
        return CATEGORY_TO_LOGO_STYLE.get(category, 'minimal')

    def _infer_neckline(self, meta: Dict) -> str:
        """Infer neckline from metadata."""
        # Check if explicitly set
        neckline = meta.get('neckline', '')
        if neckline:
            return neckline.lower()

        # Infer from category or name
        category = meta.get('category', '').lower()
        if 'polo' in category:
            return 'polo'
        if 'henley' in category:
            return 'henley'

        # Default to crew
        return 'crew'

    def _infer_pattern_density(self, meta: Dict) -> str:
        """Infer pattern density from category and visual cues."""
        category = meta.get('category', '')
        base = CATEGORY_TO_PATTERN.get(category, 'minimal')

        # Check cluster profile for hints
        cluster_profile = meta.get('cluster_profile', '')
        if 'graphics' in cluster_profile.lower():
            return 'moderate' if base == 'minimal' else 'busy'
        if 'solids' in cluster_profile.lower():
            return 'solid'

        return base

    def _classify_color_family(self, color: str) -> str:
        """Classify a color into a family."""
        if not color:
            return 'neutral'

        color_lower = color.lower()
        for family, colors in COLOR_FAMILIES.items():
            if any(c in color_lower for c in colors):
                return family
        return 'neutral'

    def _classify_color_tone(self, color: str) -> str:
        """Classify a color into a tone."""
        if not color:
            return 'neutral'

        color_lower = color.lower()
        for tone, colors in COLOR_TONES.items():
            if any(c in color_lower for c in colors):
                return tone
        return 'muted'  # Default

    def _classify_material(self, fabric: str) -> str:
        """Classify fabric into material type."""
        if not fabric:
            return 'blend'

        fabric_lower = fabric.lower()
        if 'cotton 100%' in fabric_lower or '100% cotton' in fabric_lower:
            return 'cotton'
        if 'polyester' in fabric_lower:
            return 'polyester'
        if 'wool' in fabric_lower or 'cashmere' in fabric_lower:
            return 'premium'

        return 'blend'

    # =========================================================================
    # CORE SELECTION LOGIC
    # =========================================================================

    def get_current_test_info(self, prefs: AttributeTestPreferences) -> Dict:
        """Get info about current attribute being tested."""
        phase_idx = prefs.current_phase - 1
        if phase_idx >= len(ATTRIBUTE_TEST_PHASES):
            return {'complete': True}

        phase = ATTRIBUTE_TEST_PHASES[phase_idx]
        attr_idx = prefs.current_attribute_index

        if attr_idx >= len(phase['attributes']):
            # Move to next phase
            return {'complete': True}

        current_attr = phase['attributes'][attr_idx]

        return {
            'phase': prefs.current_phase,
            'phase_name': phase['name'],
            'phase_icon': phase['icon'],
            'phase_description': phase['description'],
            'attribute': current_attr,
            'rounds_on_attr': prefs.rounds_on_current_attr,
            'rounds_needed': phase['rounds_per_attr'],
            'total_phases': len(ATTRIBUTE_TEST_PHASES),
            'complete': False,
        }

    def get_four_items(
        self,
        prefs: AttributeTestPreferences,
        candidates: List[str] = None
    ) -> Tuple[List[str], Dict]:
        """
        Select 4 items for the current attribute test.

        Returns: (items, test_info)
        - items: 4 item IDs
        - test_info: metadata about the test
        """
        if candidates is None:
            candidates = self.get_candidates(prefs)

        # Filter out already seen items
        available = [c for c in candidates if c not in prefs.seen_ids]

        if len(available) < 4:
            return available, {'error': 'not_enough_items'}

        # Get current test info
        test_info = self.get_current_test_info(prefs)
        if test_info.get('complete'):
            return [], test_info

        current_attr = test_info['attribute']
        prefs.current_test_attribute = current_attr

        # Select items based on test strategy
        if prefs.rounds_completed < 2:
            # First rounds: Pure archetype exploration
            selected = self._select_archetype_exploration(prefs, available)
        else:
            # Subsequent rounds: Attribute isolation
            selected = self._select_attribute_isolated(prefs, available, current_attr)

        # Ensure we have 4 items
        while len(selected) < 4 and available:
            remaining = [c for c in available if c not in selected]
            if remaining:
                selected.append(random.choice(remaining))
            else:
                break

        # Shuffle to avoid position bias
        random.shuffle(selected)

        # Add attribute values to test info
        test_info['items_attributes'] = {
            item_id: self.item_attributes.get(item_id, {}).get(current_attr, 'unknown')
            for item_id in selected
        }

        return selected, test_info

    def _select_archetype_exploration(
        self,
        prefs: AttributeTestPreferences,
        available: List[str]
    ) -> List[str]:
        """Select 4 items covering different archetypes."""
        selected = []
        used_archetypes = set()

        # Try to get one from each archetype
        for archetype in ATTRIBUTE_VALUES['archetype']:
            if len(selected) >= 4:
                break

            candidates = [
                item_id for item_id in available
                if self.item_attributes.get(item_id, {}).get('archetype') == archetype
                and item_id not in selected
            ]

            if candidates:
                selected.append(random.choice(candidates))
                used_archetypes.add(archetype)

        return selected

    def _select_attribute_isolated(
        self,
        prefs: AttributeTestPreferences,
        available: List[str],
        test_attr: str
    ) -> List[str]:
        """
        Select 4 items that are visually similar but differ on test_attr.

        Strategy:
        1. Find a "reference" item (taste-aligned or from preferred cluster)
        2. Find items visually similar to reference
        3. Among similar items, select 4 with different test_attr values
        """
        selected = []

        # Get reference embedding (taste vector or first liked item)
        reference = self._get_reference_embedding(prefs)

        if reference is None:
            # Fall back to random selection with contrast
            return self._select_with_contrast(available, test_attr)

        # Find visually similar items
        similar_items = self._find_similar_items(reference, available, k=100)

        # Group by test attribute value
        attr_groups = defaultdict(list)
        for item_id, similarity in similar_items:
            attr_value = self.item_attributes.get(item_id, {}).get(test_attr, 'unknown')
            if attr_value != 'unknown':
                attr_groups[attr_value].append((item_id, similarity))

        # Select one item from each attribute value (prefer higher similarity)
        attr_values = ATTRIBUTE_VALUES.get(test_attr, list(attr_groups.keys()))

        for attr_value in attr_values:
            if len(selected) >= 4:
                break

            if attr_value in attr_groups and attr_groups[attr_value]:
                # Sort by similarity, pick best
                attr_groups[attr_value].sort(key=lambda x: x[1], reverse=True)
                best_item = attr_groups[attr_value][0][0]
                selected.append(best_item)

        # If not enough variety, add from remaining groups
        for attr_value, items in attr_groups.items():
            if len(selected) >= 4:
                break
            for item_id, sim in items:
                if item_id not in selected:
                    selected.append(item_id)
                    break

        return selected

    def _get_reference_embedding(self, prefs: AttributeTestPreferences) -> Optional[np.ndarray]:
        """Get reference embedding for similarity search."""
        if prefs.taste_vector is not None:
            return prefs.taste_vector

        if prefs.reference_embedding is not None:
            return prefs.reference_embedding

        # Use first liked item
        if prefs.liked_ids:
            first_liked = prefs.liked_ids[0]
            emb = self.embeddings_data.get(first_liked, {}).get('embedding')
            if emb is not None:
                return emb / np.linalg.norm(emb)

        return None

    def _find_similar_items(
        self,
        reference: np.ndarray,
        available: List[str],
        k: int = 50
    ) -> List[Tuple[str, float]]:
        """Find items most similar to reference embedding."""
        similarities = []

        for item_id in available:
            emb = self.embeddings_data.get(item_id, {}).get('embedding')
            if emb is not None:
                emb_norm = emb / np.linalg.norm(emb)
                sim = np.dot(reference, emb_norm)
                similarities.append((item_id, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:k]

    def _select_with_contrast(
        self,
        available: List[str],
        test_attr: str
    ) -> List[str]:
        """Select 4 items maximizing contrast on test_attr."""
        selected = []
        used_values = set()

        random.shuffle(available)

        for item_id in available:
            if len(selected) >= 4:
                break

            attr_value = self.item_attributes.get(item_id, {}).get(test_attr, 'unknown')

            if attr_value not in used_values or len(used_values) >= len(ATTRIBUTE_VALUES.get(test_attr, [])):
                selected.append(item_id)
                used_values.add(attr_value)

        return selected

    # =========================================================================
    # PREFERENCE LEARNING
    # =========================================================================

    def record_choice(
        self,
        prefs: AttributeTestPreferences,
        winner_id: str,
        all_shown: List[str]
    ) -> AttributeTestPreferences:
        """
        Record user's choice and update attribute preferences.

        Learning strategy:
        1. Winner's attribute value gets positive score
        2. Losers' attribute values get negative scores
        3. After enough data, lock in the preference
        """
        losers = [item_id for item_id in all_shown if item_id != winner_id]
        current_attr = prefs.current_test_attribute

        # Get attribute values
        winner_value = self.item_attributes.get(winner_id, {}).get(current_attr, 'unknown')
        loser_values = [
            self.item_attributes.get(lid, {}).get(current_attr, 'unknown')
            for lid in losers
        ]

        # Initialize scores if needed
        if current_attr not in prefs.attribute_scores:
            prefs.attribute_scores[current_attr] = defaultdict(float)

        # Update scores
        prefs.attribute_scores[current_attr][winner_value] += 1.0
        for lv in loser_values:
            if lv != 'unknown':
                prefs.attribute_scores[current_attr][lv] -= 0.33

        # Record in history
        prefs.test_history.append({
            'round': prefs.rounds_completed,
            'attribute': current_attr,
            'winner': winner_id,
            'winner_value': winner_value,
            'losers': losers,
            'loser_values': loser_values,
        })

        # Update seen items
        prefs.liked_ids.append(winner_id)
        for loser in losers:
            prefs.disliked_ids.append(loser)

        # Update taste vector
        self._update_taste_contrastive(prefs, winner_id, losers)

        # Store reference embedding from first choice
        if prefs.reference_embedding is None:
            emb = self.embeddings_data.get(winner_id, {}).get('embedding')
            if emb is not None:
                prefs.reference_embedding = emb / np.linalg.norm(emb)
                prefs.reference_cluster = self.item_to_cluster.get(winner_id)

        # Update cluster tracking
        for item_id in all_shown:
            cluster = self.item_to_cluster.get(item_id)
            if cluster is not None:
                prefs.swipes_per_cluster[cluster] = prefs.swipes_per_cluster.get(cluster, 0) + 1
                if item_id == winner_id:
                    prefs.likes_per_cluster[cluster] = prefs.likes_per_cluster.get(cluster, 0) + 1

        # Update attribute tracking (brand, color, etc.)
        winner_meta = self.metadata.get(winner_id, {})
        for attr_type in ['color', 'fit', 'fabric', 'category']:
            attr_value = winner_meta.get(attr_type, '')
            if attr_value:
                if attr_type not in prefs.attribute_likes:
                    prefs.attribute_likes[attr_type] = {}
                prefs.attribute_likes[attr_type][attr_value] = \
                    prefs.attribute_likes[attr_type].get(attr_value, 0) + 1

        brand = winner_meta.get('brand', '')
        if brand:
            prefs.brand_likes[brand] = prefs.brand_likes.get(brand, 0) + 1

        # Increment counters
        prefs.rounds_completed += 1
        prefs.rounds_on_current_attr += 1

        # Check if we should lock this attribute and move on
        self._maybe_advance_attribute(prefs)

        return prefs

    def _update_taste_contrastive(
        self,
        prefs: AttributeTestPreferences,
        winner_id: str,
        loser_ids: List[str]
    ):
        """Update taste vector using contrastive learning."""
        # Get winner embedding
        winner_emb = self.embeddings_data.get(winner_id, {}).get('embedding')
        if winner_emb is None:
            return

        winner_emb = winner_emb / np.linalg.norm(winner_emb)

        # Get loser embeddings
        loser_embs = []
        for loser_id in loser_ids:
            emb = self.embeddings_data.get(loser_id, {}).get('embedding')
            if emb is not None:
                loser_embs.append(emb / np.linalg.norm(emb))

        if loser_embs:
            loser_mean = np.mean(loser_embs, axis=0)
            loser_mean = loser_mean / np.linalg.norm(loser_mean)

        # Update taste vector
        if prefs.taste_vector is None:
            prefs.taste_vector = winner_emb.copy()
        else:
            prefs.taste_vector = 0.7 * prefs.taste_vector + 0.3 * winner_emb
            prefs.taste_vector = prefs.taste_vector / np.linalg.norm(prefs.taste_vector)

        # Update anti-taste vector
        if loser_embs:
            if prefs.anti_taste_vector is None:
                prefs.anti_taste_vector = loser_mean.copy()
            else:
                prefs.anti_taste_vector = 0.8 * prefs.anti_taste_vector + 0.2 * loser_mean
                prefs.anti_taste_vector = prefs.anti_taste_vector / np.linalg.norm(prefs.anti_taste_vector)

        # Track history for stability
        prefs.taste_vector_history.append(prefs.taste_vector.copy())
        if len(prefs.taste_vector_history) > 5:
            prefs.taste_vector_history = prefs.taste_vector_history[-5:]

    def _maybe_advance_attribute(self, prefs: AttributeTestPreferences):
        """Check if we should lock attribute and move to next one."""
        phase_idx = prefs.current_phase - 1
        if phase_idx >= len(ATTRIBUTE_TEST_PHASES):
            return

        phase = ATTRIBUTE_TEST_PHASES[phase_idx]
        rounds_needed = phase['rounds_per_attr']

        if prefs.rounds_on_current_attr >= rounds_needed:
            # Lock in the preference for current attribute
            current_attr = prefs.current_test_attribute

            if current_attr and current_attr in prefs.attribute_scores:
                scores = prefs.attribute_scores[current_attr]
                if scores:
                    best_value = max(scores.items(), key=lambda x: x[1])[0]
                    best_score = scores[best_value]

                    # Calculate confidence (difference from second best)
                    sorted_scores = sorted(scores.values(), reverse=True)
                    if len(sorted_scores) > 1:
                        confidence = (sorted_scores[0] - sorted_scores[1]) / max(1, sorted_scores[0])
                    else:
                        confidence = 1.0

                    prefs.locked_attributes[current_attr] = best_value
                    prefs.attribute_confidence[current_attr] = confidence

            # Move to next attribute
            prefs.current_attribute_index += 1
            prefs.rounds_on_current_attr = 0

            # Check if phase is complete
            if prefs.current_attribute_index >= len(phase['attributes']):
                prefs.phase_complete.add(prefs.current_phase)
                prefs.current_phase += 1
                prefs.current_attribute_index = 0

    def record_skip_all(
        self,
        prefs: AttributeTestPreferences,
        all_shown: List[str]
    ) -> AttributeTestPreferences:
        """Record when user skips all 4 (none appealing)."""
        # Mark all as mild negative
        for item_id in all_shown:
            prefs.disliked_ids.append(item_id)

        # Update anti-taste
        all_embs = []
        for item_id in all_shown:
            emb = self.embeddings_data.get(item_id, {}).get('embedding')
            if emb is not None:
                all_embs.append(emb / np.linalg.norm(emb))

        if all_embs:
            all_mean = np.mean(all_embs, axis=0)
            all_mean = all_mean / np.linalg.norm(all_mean)

            if prefs.anti_taste_vector is None:
                prefs.anti_taste_vector = all_mean
            else:
                prefs.anti_taste_vector = 0.7 * prefs.anti_taste_vector + 0.3 * all_mean
                prefs.anti_taste_vector = prefs.anti_taste_vector / np.linalg.norm(prefs.anti_taste_vector)

        prefs.rounds_completed += 1
        prefs.rounds_on_current_attr += 1

        # Still advance attribute testing
        self._maybe_advance_attribute(prefs)

        return prefs

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    def is_session_complete(self, prefs: AttributeTestPreferences) -> bool:
        """Check if attribute testing session is complete."""
        # Not enough rounds yet
        if prefs.rounds_completed < self.MIN_ROUNDS:
            return False

        # Hard cap
        if prefs.rounds_completed >= self.MAX_ROUNDS:
            return True

        # All phases complete
        if prefs.current_phase > len(ATTRIBUTE_TEST_PHASES):
            return True

        # Check taste stability (early exit if very stable)
        if len(prefs.taste_vector_history) >= 4:
            recent = prefs.taste_vector_history[-4:]
            stabilities = [np.dot(recent[i], recent[i+1]) for i in range(len(recent)-1)]
            if np.mean(stabilities) >= 0.95:
                # Very stable taste, can exit early
                return prefs.rounds_completed >= self.MIN_ROUNDS

        return False

    def get_session_summary(self, prefs: AttributeTestPreferences) -> Dict:
        """Get comprehensive summary of attribute testing session."""
        base_summary = self.get_preference_summary(prefs)

        # Attribute test specifics
        base_summary['rounds_completed'] = prefs.rounds_completed
        base_summary['phases_completed'] = len(prefs.phase_complete)
        base_summary['total_phases'] = len(ATTRIBUTE_TEST_PHASES)
        base_summary['current_phase'] = prefs.current_phase

        # Learned attribute preferences
        learned_prefs = {}
        for attr, value in prefs.locked_attributes.items():
            learned_prefs[attr] = {
                'value': value,
                'confidence': prefs.attribute_confidence.get(attr, 0),
            }
        base_summary['learned_preferences'] = learned_prefs

        # Attribute scores (raw)
        base_summary['attribute_scores'] = {
            attr: dict(scores)
            for attr, scores in prefs.attribute_scores.items()
        }

        # Phase breakdown
        phase_info = []
        for phase in ATTRIBUTE_TEST_PHASES:
            phase_num = phase['phase']
            phase_info.append({
                'phase': phase_num,
                'name': phase['name'],
                'icon': phase['icon'],
                'complete': phase_num in prefs.phase_complete,
                'attributes': phase['attributes'],
            })
        base_summary['phases'] = phase_info

        # Information gain calculation
        # Each 4-choice gives log2(4) = 2 bits base
        # But attribute isolation gives MORE because we're learning structured info
        info_bits = prefs.rounds_completed * 2.0  # Base
        info_bits += len(prefs.locked_attributes) * 1.5  # Bonus for locked attributes
        base_summary['information_bits'] = info_bits
        base_summary['info_bits_gained'] = info_bits

        # Comparisons made
        base_summary['pairwise_comparisons'] = prefs.rounds_completed * 6  # 4 items = 6 pairs

        return base_summary

    def get_progress(self, prefs: AttributeTestPreferences) -> Dict:
        """Get visual progress information."""
        total_attrs = sum(len(p['attributes']) for p in ATTRIBUTE_TEST_PHASES)
        tested_attrs = len(prefs.locked_attributes)

        # Current position
        test_info = self.get_current_test_info(prefs)

        return {
            'percent_complete': (tested_attrs / total_attrs) * 100,
            'phases_done': len(prefs.phase_complete),
            'phases_total': len(ATTRIBUTE_TEST_PHASES),
            'attributes_learned': tested_attrs,
            'attributes_total': total_attrs,
            'rounds_done': prefs.rounds_completed,
            'rounds_max': self.MAX_ROUNDS,
            'current_phase': test_info.get('phase_name', 'Complete'),
            'current_attribute': test_info.get('attribute', 'Complete'),
            'is_complete': self.is_session_complete(prefs),
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Attribute Test Engine ===\n")

    engine = AttributeTestEngine()

    # Test item attribute computation
    print("Sample item attributes:")
    for item_id in list(engine.item_ids)[:5]:
        attrs = engine.item_attributes.get(item_id, {})
        meta = engine.metadata.get(item_id, {})
        print(f"  {item_id}:")
        print(f"    Category: {meta.get('category')}")
        print(f"    Archetype: {attrs.get('archetype')}")
        print(f"    Logo style: {attrs.get('logo_style')}")
        print(f"    Color family: {attrs.get('color_family')}")
        print()

    # Simulate a session
    prefs = AttributeTestPreferences(user_id="test_user")

    print("\n=== Simulated Session ===\n")

    for round_num in range(15):
        items, test_info = engine.get_four_items(prefs)

        if test_info.get('complete') or len(items) < 4:
            print(f"Session complete at round {round_num}")
            break

        # Print current test
        print(f"Round {round_num + 1} - Testing: {test_info.get('attribute', 'N/A')}")
        print(f"  Phase: {test_info.get('phase_name', 'N/A')}")

        # Show items and their test attribute values
        for item_id in items:
            attr_val = test_info['items_attributes'].get(item_id, '?')
            meta = engine.metadata.get(item_id, {})
            print(f"    {meta.get('category', 'N/A')[:15]:15} | {attr_val}")

        # Simulate choice (prefer first item)
        winner = items[0]
        prefs = engine.record_choice(prefs, winner, items)

        print(f"  → Chose: {engine.metadata.get(winner, {}).get('category', 'N/A')}")
        print()

        if engine.is_session_complete(prefs):
            print(f"\n=== Session complete! ===")
            break

    # Print summary
    summary = engine.get_session_summary(prefs)
    print(f"\n=== Session Summary ===")
    print(f"Rounds: {summary['rounds_completed']}")
    print(f"Phases: {summary['phases_completed']}/{summary['total_phases']}")
    print(f"Info bits: {summary['information_bits']:.1f}")
    print(f"\nLearned Preferences:")
    for attr, info in summary['learned_preferences'].items():
        print(f"  {attr}: {info['value']} (confidence: {info['confidence']:.2f})")
