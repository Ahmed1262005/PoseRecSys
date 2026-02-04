"""
Predictive Category-Focused Four-Choice Engine

KEY INSIGHT: Test ONE category at a time with items from DIFFERENT clusters.
Learn cluster preferences WITHIN each category, then PREDICT user's next choice.
When prediction accuracy is high → category is "learned" → move to next category.

Algorithm:
1. Focus on one category (e.g., Graphics T-shirts)
2. Show 4 items from 4 DIFFERENT visual clusters within that category
3. User picks one → learn cluster preference for this category
4. PREDICT next choice before showing
5. If predictions are accurate with high confidence → category complete
6. Move to next category

This validates learning: if we can predict, we've learned.

Gender Support:
- gender="male" uses men's t-shirt data (HPImages)
- gender="female" uses women's fashion data (womenClothes)
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import random

from .swipe_engine import SwipeEngine, UserPreferences
from gender_config import (
    get_config, get_category_order, get_attribute_prompts,
    get_attribute_weights, get_embeddings_path, get_attributes_path,
    MENS_CONFIG, WOMENS_CONFIG
)

# For backward compatibility, also import from attribute_classifier
try:
    from attribute_classifier import ATTRIBUTE_PROMPTS as MENS_ATTRIBUTE_PROMPTS
    from attribute_classifier import ATTRIBUTE_WEIGHTS as MENS_ATTRIBUTE_WEIGHTS
    from attribute_classifier import load_item_attributes
except ImportError:
    MENS_ATTRIBUTE_PROMPTS = MENS_CONFIG['attribute_prompts']
    MENS_ATTRIBUTE_WEIGHTS = MENS_CONFIG['attribute_weights']
    load_item_attributes = None

try:
    from women_attribute_classifier import (
        WOMEN_ATTRIBUTE_PROMPTS, WOMEN_ATTRIBUTE_WEIGHTS,
        load_women_attributes
    )
except ImportError:
    WOMEN_ATTRIBUTE_PROMPTS = WOMENS_CONFIG['attribute_prompts']
    WOMEN_ATTRIBUTE_WEIGHTS = WOMENS_CONFIG['attribute_weights']
    load_women_attributes = None


# Default category orders (from gender_config)
MENS_CATEGORY_ORDER = MENS_CONFIG['category_order']
WOMENS_CATEGORY_ORDER = WOMENS_CONFIG['category_order']

# Minimum clusters per category for valid testing
MIN_CLUSTERS_PER_CATEGORY = 4


@dataclass
class PredictivePreferences(UserPreferences):
    """Preferences for predictive category-focused learning."""

    # Gender for this session
    gender: str = "male"

    # Category progression
    category_order: List[str] = field(default_factory=list)
    current_category_index: int = 0
    completed_categories: Set[str] = field(default_factory=set)

    # Per-category cluster preferences: category -> cluster -> score
    category_cluster_scores: Dict[str, Dict[int, float]] = field(default_factory=dict)
    category_cluster_counts: Dict[str, Dict[int, int]] = field(default_factory=dict)

    # Prediction tracking
    predictions: List[Dict] = field(default_factory=list)  # {predicted, actual, correct, confidence}
    consecutive_correct: int = 0
    last_prediction: Dict = field(default_factory=dict)

    # Round tracking
    rounds_per_category: Dict[str, int] = field(default_factory=dict)
    rounds_completed: int = 0

    # Current items shown (for validation)
    current_four: List[str] = field(default_factory=list)
    current_clusters: List[int] = field(default_factory=list)

    # NEW: Attribute-level preference tracking
    # attribute_wins[attr_name][value] = number of times this value won
    attribute_wins: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # attribute_counts[attr_name][value] = number of times this value was shown
    attribute_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def current_category(self) -> Optional[str]:
        if self.current_category_index < len(self.category_order):
            return self.category_order[self.current_category_index]
        return None


class PredictiveFourEngine(SwipeEngine):
    """
    Predictive Category-Focused Four-Choice Engine.

    Strategy:
    1. Focus on ONE category at a time
    2. Show 4 items from 4 DIFFERENT clusters within category
    3. Track cluster preferences per category
    4. Predict next choice, validate prediction
    5. When predictions are accurate → category complete

    Gender Support:
    - gender="male" uses men's t-shirt embeddings and attributes
    - gender="female" uses women's fashion embeddings and attributes
    """

    # Stopping criteria
    MIN_ROUNDS_PER_CATEGORY = 4
    MAX_ROUNDS_PER_CATEGORY = 12
    REQUIRED_CONSECUTIVE_CORRECT = 3  # Need 3 correct predictions to complete
    MIN_CONFIDENCE_THRESHOLD = 0.15   # Minimum confidence gap

    def __init__(
        self,
        gender: str = "male",
        embeddings_path: str = None,
        attributes_path: str = None,
        **kwargs
    ):
        """
        Initialize the engine with gender-specific data.

        Args:
            gender: "male" or "female" (also accepts "men", "women", etc.)
            embeddings_path: Override path to embeddings (optional)
            attributes_path: Override path to attributes (optional)
            **kwargs: Additional args passed to SwipeEngine
        """
        # Normalize gender
        self.gender = self._normalize_gender(gender)

        # Get gender-specific configuration
        self.config = get_config(self.gender)

        # Set default paths based on gender if not provided
        if embeddings_path is None:
            embeddings_path = str(self.config['embeddings_path'])
        if attributes_path is None:
            attributes_path = str(self.config['attributes_path'])

        # Get gender-specific prompts and weights
        if self.gender == "female":
            self.attribute_prompts = WOMEN_ATTRIBUTE_PROMPTS
            self.attribute_weights = WOMEN_ATTRIBUTE_WEIGHTS
            self.default_category_order = WOMENS_CATEGORY_ORDER
        else:
            self.attribute_prompts = MENS_ATTRIBUTE_PROMPTS
            self.attribute_weights = MENS_ATTRIBUTE_WEIGHTS
            self.default_category_order = MENS_CATEGORY_ORDER

        # Initialize base engine with embeddings path
        super().__init__(embeddings_path=embeddings_path, **kwargs)

        # For women's fashion, load full metadata including brand from women_items.pkl
        if self.gender == "female":
            self._load_women_metadata()

        print(f"\nGender-Aware Engine initialized:")
        print(f"  Gender: {self.gender}")
        print(f"  Embeddings: {embeddings_path}")
        print(f"  Attributes: {attributes_path}")
        print(f"  Categories: {self.default_category_order}")

        # Build category-cluster index
        self._build_category_cluster_index()

        # Load pre-computed item attributes
        self._load_item_attributes(attributes_path)

    @staticmethod
    def _normalize_gender(gender: str) -> str:
        """Normalize gender string to 'male' or 'female'."""
        gender = gender.lower().strip()
        if gender in ("female", "women", "woman", "f", "w"):
            return "female"
        return "male"

    def _load_women_metadata(self):
        """Load full metadata for women's fashion from women_items.pkl.

        The women_items.pkl contains full product metadata including brand,
        which is not present in women_embeddings.pkl.
        """
        women_items_path = Path("/home/ubuntu/recSys/outfitTransformer/data/women_fashion/processed/women_items.pkl")

        if not women_items_path.exists():
            print(f"  Warning: Women's metadata file not found at {women_items_path}")
            return

        try:
            with open(women_items_path, 'rb') as f:
                women_data = pickle.load(f)

            items = women_data.get('items', {})
            loaded_count = 0
            brand_count = 0

            # Merge women_items metadata into self.metadata
            for item_id, item_data in items.items():
                if item_id in self.item_ids:
                    # Create/update metadata entry
                    if item_id not in self.metadata:
                        self.metadata[item_id] = {}

                    # Copy all useful fields from women_items
                    self.metadata[item_id].update({
                        'brand': item_data.get('brand', ''),
                        'color': item_data.get('color', ''),
                        'fit': item_data.get('fit', ''),
                        'neckline': item_data.get('neckline', ''),
                        'fabric': item_data.get('fabric', ''),
                        'pattern': item_data.get('pattern', ''),
                        'category': item_data.get('category', ''),
                        'subcategory': item_data.get('subcategory', ''),
                        'url': item_data.get('url', ''),
                    })
                    loaded_count += 1
                    if item_data.get('brand'):
                        brand_count += 1

            print(f"  Loaded women's metadata for {loaded_count} items ({brand_count} with brand)")

            # Re-index metadata to capture new brands
            self._index_metadata()

        except Exception as e:
            print(f"  Warning: Could not load women's metadata: {e}")

    def _build_category_cluster_index(self):
        """Build index of which clusters contain items from which categories."""
        self.category_clusters = defaultdict(set)  # category -> set of clusters
        self.cluster_category_items = defaultdict(lambda: defaultdict(list))  # cluster -> category -> items

        for item_id in self.item_ids:
            # Try to get category from metadata (CSV), fall back to embeddings data
            meta = self.metadata.get(item_id, {})
            category = meta.get('category', '')

            # If no category from CSV metadata, try embeddings data (for women's fashion)
            if not category:
                emb_data = self.embeddings_data.get(item_id, {})
                category = emb_data.get('category', '')

            cluster = self.item_to_cluster.get(item_id)

            if category and cluster is not None:
                self.category_clusters[category].add(cluster)
                self.cluster_category_items[cluster][category].append(item_id)

        # Log stats
        print("\nCategory-Cluster Distribution:")
        for category, clusters in sorted(self.category_clusters.items()):
            print(f"  {category}: {len(clusters)} clusters")

    def _load_item_attributes(self, attributes_path: str = None):
        """Load pre-computed item attributes from CLIP classification."""
        if attributes_path is None:
            # Default path
            base_dir = Path(__file__).parent.parent
            attributes_path = base_dir / "models" / "item_attributes.pkl"

        self.item_attributes = {}

        try:
            if Path(attributes_path).exists():
                self.item_attributes = load_item_attributes(str(attributes_path))
                print(f"  Loaded attributes for {len(self.item_attributes)} items")

                # Log attribute distribution sample
                if self.item_attributes:
                    sample_item = next(iter(self.item_attributes.values()))
                    print(f"  Attribute dimensions: {list(sample_item.keys())}")
            else:
                print(f"  Warning: Attributes file not found at {attributes_path}")
        except Exception as e:
            print(f"  Warning: Could not load item attributes: {e}")

    def initialize_session(
        self,
        prefs: PredictivePreferences,
        selected_categories: List[str] = None
    ) -> PredictivePreferences:
        """Initialize session with category order."""

        # Set gender on preferences
        prefs.gender = self.gender

        if selected_categories:
            # Use user's selected categories
            prefs.category_order = [c for c in selected_categories
                                    if c in self.category_clusters]
        else:
            # Use default order, filtered by available categories
            prefs.category_order = [c for c in self.default_category_order
                                    if c in self.category_clusters]

        # Initialize tracking structures
        for category in prefs.category_order:
            prefs.category_cluster_scores[category] = defaultdict(float)
            prefs.category_cluster_counts[category] = defaultdict(int)
            prefs.rounds_per_category[category] = 0

        # Initialize attribute tracking structures (gender-aware)
        for attr_name in self.attribute_prompts.keys():
            prefs.attribute_wins[attr_name] = defaultdict(int)
            prefs.attribute_counts[attr_name] = defaultdict(int)

        return prefs

    def _item_passes_onboarding_filters(self, item_id: str, prefs: PredictivePreferences) -> bool:
        """Check if an item passes all onboarding hard filters.

        Filters:
        - colors_to_avoid: exclude if item color matches any avoided color
        - materials_to_avoid: exclude if item fabric matches any avoided material
        - brands_to_avoid: exclude if item brand is in avoided brands list
        """
        info = self.get_item_info(item_id)

        # Check colors to avoid
        item_color = (info.get('color', '') or '').lower()
        if prefs.colors_to_avoid and any(c in item_color for c in prefs.colors_to_avoid):
            return False

        # Check materials to avoid
        item_fabric = (info.get('fabric', '') or '').lower()
        if prefs.materials_to_avoid and any(m in item_fabric for m in prefs.materials_to_avoid):
            return False

        # Check brands to avoid
        item_brand = (info.get('brand', '') or '').lower()
        if prefs.brands_to_avoid and item_brand and item_brand in prefs.brands_to_avoid:
            return False

        return True

    def _is_preferred_brand(self, item_id: str, prefs: PredictivePreferences) -> bool:
        """Check if an item is from a preferred brand."""
        if not prefs.preferred_brands:
            return False
        info = self.get_item_info(item_id)
        item_brand = (info.get('brand', '') or '').lower()
        return item_brand and item_brand in prefs.preferred_brands

    def _select_item_from_cluster_with_brand_priority(
        self,
        eligible_items: List[str],
        prefs: PredictivePreferences,
        already_selected: List[str]
    ) -> Optional[str]:
        """
        Select an item from eligible items, prioritizing preferred brands.

        Priority order:
        1. Items from preferred brands (if taste vector available, prefer closest to taste)
        2. Items closest to taste vector (if available)
        3. Random selection

        This ensures that when user has preferred brands, the Tinder test
        shows more items from those brands to learn their style properly.
        """
        # Remove already selected items
        eligible = [i for i in eligible_items if i not in already_selected]
        if not eligible:
            return None

        # Separate preferred brand items from others
        preferred_brand_items = [i for i in eligible if self._is_preferred_brand(i, prefs)]
        other_items = [i for i in eligible if not self._is_preferred_brand(i, prefs)]

        # Try preferred brand items first (with 70% probability if available)
        # This ensures good brand coverage while still showing variety
        use_preferred = preferred_brand_items and (not other_items or random.random() < 0.7)

        pool = preferred_brand_items if use_preferred else (other_items if other_items else eligible)

        # Select from pool
        if prefs.taste_vector is not None:
            # Prefer item closest to taste vector
            return self._select_taste_aligned(pool, prefs.taste_vector)
        else:
            # Random selection
            return random.choice(pool)

    def get_four_items(
        self,
        prefs: PredictivePreferences
    ) -> Tuple[List[str], Dict]:
        """
        Get 4 items from SAME category but DIFFERENT clusters.

        Returns: (items, test_info)
        """
        current_category = prefs.current_category

        if current_category is None:
            return [], {'status': 'all_complete', 'message': 'All categories completed'}

        # Get available clusters for this category
        available_clusters = list(self.category_clusters.get(current_category, set()))

        # Filter out clusters with no unseen AND eligible items
        valid_clusters = []
        for cluster in available_clusters:
            cluster_items = self.cluster_category_items[cluster].get(current_category, [])
            # Filter by seen_ids AND onboarding hard filters
            eligible = [i for i in cluster_items
                        if i not in prefs.seen_ids
                        and self._item_passes_onboarding_filters(i, prefs)]
            if eligible:
                valid_clusters.append(cluster)

        if len(valid_clusters) < 4:
            # Not enough clusters, try to complete with what we have or skip category
            if len(valid_clusters) < 2:
                # Skip to next category
                prefs.completed_categories.add(current_category)
                prefs.current_category_index += 1
                return self.get_four_items(prefs)

        # Select 4 different clusters
        # Strategy: Mix explored and unexplored clusters
        selected_clusters = self._select_clusters_for_test(prefs, valid_clusters, current_category)

        # Select one item from each cluster
        selected_items = []
        actual_clusters = []

        # Exclude items from previous round to prevent repetition
        previous_items = set(prefs.current_four) if prefs.current_four else set()

        for cluster in selected_clusters:
            cluster_items = self.cluster_category_items[cluster].get(current_category, [])
            # Filter out seen items, previous round items, already selected, AND apply onboarding filters
            eligible = [i for i in cluster_items
                        if i not in prefs.seen_ids
                        and i not in previous_items
                        and i not in selected_items
                        and self._item_passes_onboarding_filters(i, prefs)]

            if eligible:
                # Select item with brand priority + taste alignment
                # This ensures preferred brands are well-represented in the test
                item = self._select_item_from_cluster_with_brand_priority(
                    eligible, prefs, selected_items
                )

                if item:
                    selected_items.append(item)
                    actual_clusters.append(cluster)

            if len(selected_items) >= 4:
                break

        # Shuffle to avoid position bias
        combined = list(zip(selected_items, actual_clusters))
        random.shuffle(combined)
        selected_items, actual_clusters = zip(*combined) if combined else ([], [])
        selected_items = list(selected_items)
        actual_clusters = list(actual_clusters)

        # Store for validation
        prefs.current_four = selected_items
        prefs.current_clusters = actual_clusters

        # Make prediction BEFORE showing to user
        prediction = self._make_prediction(prefs, actual_clusters, current_category)
        prefs.last_prediction = prediction

        # Build test info
        test_info = {
            'category': current_category,
            'category_index': prefs.current_category_index + 1,
            'total_categories': len(prefs.category_order),
            'round_in_category': prefs.rounds_per_category.get(current_category, 0) + 1,
            'clusters_shown': actual_clusters,
            'prediction': prediction,
            'categories_completed': list(prefs.completed_categories),
        }

        return selected_items, test_info

    def _select_clusters_for_test(
        self,
        prefs: PredictivePreferences,
        valid_clusters: List[int],
        category: str
    ) -> List[int]:
        """
        Select 4 clusters for testing.

        Strategy:
        - Round 1-2: Diverse exploration (spread across clusters)
        - Round 3+: Include predicted-preferred cluster + diverse others
        """
        rounds_done = prefs.rounds_per_category.get(category, 0)
        scores = prefs.category_cluster_scores.get(category, {})

        if rounds_done < 2 or not scores:
            # Early exploration: random diverse selection
            random.shuffle(valid_clusters)
            return valid_clusters[:4]

        # Later rounds: include high-scoring clusters + explore others
        selected = []

        # Sort clusters by score (descending)
        scored_clusters = [(c, scores.get(c, 0.5)) for c in valid_clusters]
        scored_clusters.sort(key=lambda x: x[1], reverse=True)

        # Add top 2 scoring clusters
        for cluster, score in scored_clusters[:2]:
            if cluster not in selected:
                selected.append(cluster)

        # Add 2 less-explored clusters for diversity
        counts = prefs.category_cluster_counts.get(category, {})
        unexplored = [(c, counts.get(c, 0)) for c in valid_clusters if c not in selected]
        unexplored.sort(key=lambda x: x[1])  # Least explored first

        for cluster, count in unexplored:
            if len(selected) >= 4:
                break
            selected.append(cluster)

        # Fill remaining with random
        remaining = [c for c in valid_clusters if c not in selected]
        random.shuffle(remaining)
        for c in remaining:
            if len(selected) >= 4:
                break
            selected.append(c)

        return selected[:4]

    def _select_taste_aligned(self, items: List[str], taste_vector: np.ndarray) -> str:
        """Select item most aligned with taste vector."""
        best_item = items[0]
        best_sim = -1

        for item_id in items:
            emb = self.embeddings_data.get(item_id, {}).get('embedding')
            if emb is not None:
                emb_norm = emb / np.linalg.norm(emb)
                sim = np.dot(emb_norm, taste_vector)
                if sim > best_sim:
                    best_sim = sim
                    best_item = item_id

        return best_item

    def _make_prediction(
        self,
        prefs: PredictivePreferences,
        clusters: List[int],
        category: str
    ) -> Dict:
        """
        Predict which item user will pick using ATTRIBUTE-BASED scoring.

        Combines:
        - Attribute scores (pattern, style, color_family, fit_vibe) - 70% weight
        - Cluster scores (legacy) - 20% weight
        - Taste vector similarity - 10% weight

        Falls back to cluster-only prediction in early rounds.
        """
        rounds_done = prefs.rounds_per_category.get(category, 0)

        # Cluster scores (legacy)
        cat_scores = prefs.category_cluster_scores.get(category, {})
        cat_counts = prefs.category_cluster_counts.get(category, {})

        cluster_scores = {}
        for cluster in clusters:
            wins = cat_scores.get(cluster, 0)
            total = cat_counts.get(cluster, 0)
            cluster_scores[cluster] = (wins + 1) / (total + 2) if total > 0 else 0.5

        # Get current items being shown
        current_items = prefs.current_four
        current_clusters = prefs.current_clusters

        if not current_items or len(current_items) < len(clusters):
            # Fallback to cluster-only prediction
            cluster_list = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)
            if cluster_list:
                return {
                    'predicted_cluster': cluster_list[0][0],
                    'confidence': round(cluster_list[0][1] - cluster_list[1][1], 3) if len(cluster_list) > 1 else 0,
                    'top_score': round(cluster_list[0][1], 3),
                    'cluster_scores': [(c, round(s, 3)) for c, s in cluster_list],
                    'has_prediction': rounds_done > 0,
                    'method': 'cluster_only'
                }
            return {'predicted_cluster': None, 'confidence': 0, 'scores': [], 'method': 'none'}

        # Compute attribute scores if we have enough data
        attr_scores = self._compute_attribute_scores(prefs)
        has_attr_data = any(
            len(scores) > 0 for scores in attr_scores.values()
        )

        # Score each item
        item_scores = []
        for i, item_id in enumerate(current_items):
            cluster = current_clusters[i] if i < len(current_clusters) else None

            # Cluster score (20% weight after round 1)
            cluster_score = cluster_scores.get(cluster, 0.5) if cluster else 0.5

            # Attribute score (70% weight)
            if has_attr_data and self.item_attributes:
                attr_score = self._get_item_attribute_score(item_id, attr_scores)
            else:
                attr_score = 0.5

            # Taste vector similarity (10% weight)
            taste_sim = 0.5
            if prefs.taste_vector is not None:
                emb = self.embeddings_data.get(item_id, {}).get('embedding')
                if emb is not None:
                    emb_norm = emb / np.linalg.norm(emb)
                    taste_sim = float(np.dot(emb_norm, prefs.taste_vector))
                    taste_sim = (taste_sim + 1) / 2  # Normalize to 0-1

            # Combined score
            # Early rounds: more cluster weight
            # Later rounds: more attribute weight
            if rounds_done < 2:
                combined = 0.5 * cluster_score + 0.3 * attr_score + 0.2 * taste_sim
            else:
                combined = 0.2 * cluster_score + 0.6 * attr_score + 0.2 * taste_sim

            item_scores.append({
                'item_id': item_id,
                'cluster': cluster,
                'cluster_score': cluster_score,
                'attr_score': attr_score,
                'taste_sim': taste_sim,
                'combined': combined
            })

        # Sort by combined score
        item_scores.sort(key=lambda x: x['combined'], reverse=True)

        predicted_item = item_scores[0]
        predicted_cluster = predicted_item['cluster']

        # Confidence = gap between top and second
        confidence = 0
        if len(item_scores) > 1:
            confidence = predicted_item['combined'] - item_scores[1]['combined']

        return {
            'predicted_cluster': predicted_cluster,
            'predicted_item': predicted_item['item_id'],
            'confidence': round(confidence, 3),
            'top_score': round(predicted_item['combined'], 3),
            'cluster_scores': [(s['cluster'], round(s['cluster_score'], 3)) for s in item_scores],
            'attr_scores_top': {
                'attr': round(predicted_item['attr_score'], 3),
                'taste': round(predicted_item['taste_sim'], 3)
            },
            'has_prediction': rounds_done > 0,
            'method': 'attribute_hybrid' if has_attr_data else 'cluster_only'
        }

    def record_choice(
        self,
        prefs: PredictivePreferences,
        winner_id: str
    ) -> Tuple[PredictivePreferences, Dict]:
        """
        Record user's choice and validate prediction.

        Returns: (updated_prefs, result_info)
        """
        current_category = prefs.current_category
        current_four = prefs.current_four
        current_clusters = prefs.current_clusters

        if winner_id not in current_four:
            return prefs, {'error': 'Invalid winner_id'}

        winner_idx = current_four.index(winner_id)
        winner_cluster = current_clusters[winner_idx]

        # Record all shown items as seen
        for item_id in current_four:
            if item_id == winner_id:
                prefs.liked_ids.append(item_id)
            else:
                prefs.disliked_ids.append(item_id)

        # Update cluster scores for this category
        for i, cluster in enumerate(current_clusters):
            prefs.category_cluster_counts[current_category][cluster] += 1
            if cluster == winner_cluster:
                prefs.category_cluster_scores[current_category][cluster] += 1

        # Update cluster tracking (for base engine compatibility)
        for cluster in current_clusters:
            prefs.swipes_per_cluster[cluster] = prefs.swipes_per_cluster.get(cluster, 0) + 1
            if cluster == winner_cluster:
                prefs.likes_per_cluster[cluster] = prefs.likes_per_cluster.get(cluster, 0) + 1
            else:
                prefs.dislikes_per_cluster[cluster] = prefs.dislikes_per_cluster.get(cluster, 0) + 1

        # Update taste vectors
        loser_ids = [i for i in current_four if i != winner_id]
        self._update_taste_contrastive(prefs, winner_id, loser_ids)

        # Update attribute preferences (NEW)
        self._update_attribute_preferences(prefs, winner_id, loser_ids)

        # Validate prediction
        prediction = prefs.last_prediction
        predicted_cluster = prediction.get('predicted_cluster')
        prediction_correct = (predicted_cluster == winner_cluster) if predicted_cluster is not None else False
        had_prediction = prediction.get('has_prediction', False)

        # Track prediction accuracy
        if had_prediction:
            prefs.predictions.append({
                'category': current_category,
                'predicted_cluster': predicted_cluster,
                'actual_cluster': winner_cluster,
                'correct': prediction_correct,
                'confidence': prediction.get('confidence', 0)
            })

            if prediction_correct:
                prefs.consecutive_correct += 1
            else:
                prefs.consecutive_correct = 0

        # Update round counters
        prefs.rounds_per_category[current_category] += 1
        prefs.rounds_completed += 1

        # Check if category is complete
        category_complete = self._check_category_complete(prefs, current_category)

        result_info = {
            'winner_cluster': winner_cluster,
            'prediction_correct': prediction_correct,
            'consecutive_correct': prefs.consecutive_correct,
            'confidence': prediction.get('confidence', 0),
            'category_complete': category_complete,
            'had_prediction': had_prediction,
        }

        if category_complete:
            prefs.completed_categories.add(current_category)
            prefs.current_category_index += 1
            prefs.consecutive_correct = 0  # Reset for next category

            result_info['next_category'] = prefs.current_category
            result_info['all_complete'] = prefs.current_category is None

        return prefs, result_info

    def _update_taste_contrastive(
        self,
        prefs: PredictivePreferences,
        winner_id: str,
        loser_ids: List[str]
    ):
        """Update taste vector using contrastive learning."""
        winner_emb = self.embeddings_data.get(winner_id, {}).get('embedding')
        if winner_emb is None:
            return

        winner_emb = winner_emb / np.linalg.norm(winner_emb)

        loser_embs = []
        for loser_id in loser_ids:
            emb = self.embeddings_data.get(loser_id, {}).get('embedding')
            if emb is not None:
                loser_embs.append(emb / np.linalg.norm(emb))

        # Update taste vector
        if prefs.taste_vector is None:
            prefs.taste_vector = winner_emb.copy()
        else:
            prefs.taste_vector = 0.7 * prefs.taste_vector + 0.3 * winner_emb
            prefs.taste_vector = prefs.taste_vector / np.linalg.norm(prefs.taste_vector)

        # Update anti-taste vector
        if loser_embs:
            loser_mean = np.mean(loser_embs, axis=0)
            loser_mean = loser_mean / np.linalg.norm(loser_mean)

            if prefs.anti_taste_vector is None:
                prefs.anti_taste_vector = loser_mean
            else:
                prefs.anti_taste_vector = 0.8 * prefs.anti_taste_vector + 0.2 * loser_mean
                prefs.anti_taste_vector = prefs.anti_taste_vector / np.linalg.norm(prefs.anti_taste_vector)

        # Track history
        prefs.taste_vector_history.append(prefs.taste_vector.copy())
        if len(prefs.taste_vector_history) > 5:
            prefs.taste_vector_history = prefs.taste_vector_history[-5:]

    def _update_attribute_preferences(
        self,
        prefs: PredictivePreferences,
        winner_id: str,
        loser_ids: List[str]
    ):
        """
        Update attribute-level preferences based on user's choice.

        For each attribute dimension (pattern, style, color_family, fit_vibe):
        - Winner's attribute value gets a win
        - All shown attribute values get counted
        """
        if not self.item_attributes:
            return

        winner_attrs = self.item_attributes.get(winner_id, {})
        if not winner_attrs:
            return

        # Update for each attribute dimension (gender-aware)
        for attr_name in self.attribute_prompts.keys():
            winner_val = winner_attrs.get(attr_name)
            if winner_val:
                # Winner gets a win
                prefs.attribute_wins[attr_name][winner_val] += 1
                prefs.attribute_counts[attr_name][winner_val] += 1

            # Losers get counted (but no win)
            for loser_id in loser_ids:
                loser_attrs = self.item_attributes.get(loser_id, {})
                loser_val = loser_attrs.get(attr_name)
                if loser_val:
                    prefs.attribute_counts[attr_name][loser_val] += 1

    def _compute_attribute_scores(
        self,
        prefs: PredictivePreferences
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute Bayesian preference scores for each attribute value.

        Returns: {attr_name: {value: score}}
        where score = (wins + 1) / (count + 2) (Laplace smoothing)
        """
        scores = {}
        for attr_name in self.attribute_prompts.keys():
            scores[attr_name] = {}
            counts = prefs.attribute_counts.get(attr_name, {})
            wins = prefs.attribute_wins.get(attr_name, {})

            for value, count in counts.items():
                if count > 0:
                    win_count = wins.get(value, 0)
                    scores[attr_name][value] = (win_count + 1) / (count + 2)
                else:
                    scores[attr_name][value] = 0.5  # Prior

        return scores

    def _get_item_attribute_score(
        self,
        item_id: str,
        attr_scores: Dict[str, Dict[str, float]]
    ) -> float:
        """
        Compute weighted attribute score for an item.

        Uses gender-specific attribute_weights to combine scores across dimensions.
        """
        item_attrs = self.item_attributes.get(item_id, {})
        if not item_attrs:
            return 0.5  # Default score

        score = 0.0
        total_weight = 0.0

        for attr_name, weight in self.attribute_weights.items():
            if attr_name == 'cluster':
                continue  # Skip cluster weight for attribute-only score

            attr_val = item_attrs.get(attr_name)
            if attr_val and attr_name in attr_scores:
                score += weight * attr_scores[attr_name].get(attr_val, 0.5)
                total_weight += weight

        if total_weight > 0:
            return score / total_weight
        return 0.5

    def _check_category_complete(
        self,
        prefs: PredictivePreferences,
        category: str
    ) -> bool:
        """
        Check if category learning is complete using ADAPTIVE STOPPING.

        NEW criteria based on attribute confidence:
        1. MIN_ROUNDS + consecutive correct predictions → learned!
        2. MIN_ROUNDS + high attribute confidence (3+ attributes clear) → learned!
        3. MIN_ROUNDS + clear cluster preference emerging → learned!
        4. MAX_ROUNDS reached → move on
        """
        rounds_done = prefs.rounds_per_category.get(category, 0)

        # Need minimum rounds first
        if rounds_done < self.MIN_ROUNDS_PER_CATEGORY:
            return False

        # Success case 1: Consecutive correct predictions
        if prefs.consecutive_correct >= self.REQUIRED_CONSECUTIVE_CORRECT:
            return True

        # Success case 2 (NEW): Attribute confidence check
        attr_scores = self._compute_attribute_scores(prefs)
        confident_attrs = 0

        for attr_name, scores in attr_scores.items():
            if not scores:
                continue

            # Sort scores to find gap between top and others
            sorted_scores = sorted(scores.values(), reverse=True)
            if len(sorted_scores) >= 2:
                gap = sorted_scores[0] - sorted_scores[1]
                # Clear preference if gap >= 0.20 (70% vs 50% for example)
                if gap >= 0.20:
                    confident_attrs += 1
                # Also count if we have a strong preference (>0.6) or strong avoidance (<0.35)
                elif sorted_scores[0] >= 0.65 or sorted_scores[-1] <= 0.30:
                    confident_attrs += 0.5

        # Exit if confident on 3+ attribute dimensions
        if confident_attrs >= 3:
            return True

        # Success case 3: Clear cluster preference emerged (legacy)
        scores = prefs.category_cluster_scores.get(category, {})
        counts = prefs.category_cluster_counts.get(category, {})
        if scores and counts:
            cluster_prefs = []
            for cluster, count in counts.items():
                if count >= 2:
                    wins = scores.get(cluster, 0)
                    pref_score = (wins + 1) / (count + 2)
                    cluster_prefs.append(pref_score)

            if len(cluster_prefs) >= 2:
                cluster_prefs.sort(reverse=True)
                if cluster_prefs[0] - cluster_prefs[1] >= 0.25:
                    return True

        # Fallback: Max rounds reached
        if rounds_done >= self.MAX_ROUNDS_PER_CATEGORY:
            return True

        return False

    def record_skip_all(
        self,
        prefs: PredictivePreferences
    ) -> PredictivePreferences:
        """Record when user skips all 4 (none appealing)."""
        current_four = prefs.current_four
        current_category = prefs.current_category

        # Mark all as disliked
        for item_id in current_four:
            prefs.disliked_ids.append(item_id)

        # Update anti-taste
        all_embs = []
        for item_id in current_four:
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

        # Increment rounds
        prefs.rounds_per_category[current_category] = prefs.rounds_per_category.get(current_category, 0) + 1
        prefs.rounds_completed += 1
        prefs.consecutive_correct = 0  # Reset

        return prefs

    def is_session_complete(self, prefs: PredictivePreferences) -> bool:
        """Check if all categories are complete."""
        return prefs.current_category is None

    def get_session_summary(self, prefs: PredictivePreferences) -> Dict:
        """Get comprehensive session summary."""
        base_summary = self.get_preference_summary(prefs)

        # Add gender
        base_summary['gender'] = prefs.gender

        # Add predictive-specific stats
        base_summary['rounds_completed'] = prefs.rounds_completed
        base_summary['categories_completed'] = list(prefs.completed_categories)
        base_summary['total_categories'] = len(prefs.category_order)
        base_summary['current_category'] = prefs.current_category

        # Prediction accuracy
        correct_predictions = sum(1 for p in prefs.predictions if p.get('correct'))
        total_predictions = len(prefs.predictions)
        base_summary['prediction_accuracy'] = (
            round(correct_predictions / total_predictions, 3)
            if total_predictions > 0 else 0
        )
        base_summary['total_predictions'] = total_predictions
        base_summary['correct_predictions'] = correct_predictions

        # Per-category breakdown
        category_stats = {}
        for category in prefs.category_order:
            rounds = prefs.rounds_per_category.get(category, 0)
            scores = prefs.category_cluster_scores.get(category, {})
            counts = prefs.category_cluster_counts.get(category, {})

            # Find preferred clusters
            cluster_prefs = []
            for cluster, count in counts.items():
                wins = scores.get(cluster, 0)
                if count > 0:
                    pref_score = (wins + 1) / (count + 2)
                    cluster_prefs.append({
                        'cluster': cluster,
                        'score': round(pref_score, 3),
                        'wins': wins,
                        'total': count,
                        'profile': self._get_cluster_profile(cluster).get('description', '')
                    })

            cluster_prefs.sort(key=lambda x: x['score'], reverse=True)

            category_stats[category] = {
                'rounds': rounds,
                'complete': category in prefs.completed_categories,
                'cluster_preferences': cluster_prefs[:3],  # Top 3
            }

        base_summary['category_stats'] = category_stats

        # Information bits
        base_summary['information_bits'] = prefs.rounds_completed * 2.0
        base_summary['pairwise_comparisons'] = prefs.rounds_completed * 6

        # NEW: Attribute preferences summary
        attr_scores = self._compute_attribute_scores(prefs)
        attribute_preferences = {}

        for attr_name, scores in attr_scores.items():
            if not scores:
                continue

            # Sort by score
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            # Find preferred (>0.55) and avoided (<0.35)
            preferred = [(v, round(s, 3)) for v, s in sorted_items if s >= 0.55]
            avoided = [(v, round(s, 3)) for v, s in sorted_items if s <= 0.35]

            attribute_preferences[attr_name] = {
                'preferred': preferred,
                'avoided': avoided,
                'all_scores': [(v, round(s, 3)) for v, s in sorted_items]
            }

        base_summary['attribute_preferences'] = attribute_preferences

        return base_summary

    def _map_tinder_to_onboarding_category(self, tinder_category: str) -> str:
        """Map tinder test category to onboarding category name for preference lookup."""
        mapping = {
            'tops_knitwear': 'tops',
            'tops_woven': 'tops',
            'tops_sleeveless': 'tops',
            'tops_special': 'tops',
            'bottoms_trousers': 'bottoms',
            'bottoms_skorts': 'skirts',
            'dresses': 'dresses',
            'outerwear': 'outerwear',
            'sportswear': 'tops',
        }
        return mapping.get(tinder_category, 'tops')

    def get_feed_preview(
        self,
        prefs: PredictivePreferences,
        items_per_category: int = 8
    ) -> Dict[str, List[Dict]]:
        """
        Generate a preview feed FILTERED BY LEARNED ATTRIBUTES.

        NEW algorithm:
        1. Compute attribute preference scores
        2. Identify preferred and avoided attribute values
        3. Filter items that match preferred attributes
        4. Exclude items with avoided attributes
        5. Rank by combined attribute + taste similarity score
        """
        seen_ids = set(prefs.liked_ids) | set(prefs.disliked_ids)
        result = {}

        # Compute attribute scores
        attr_scores = self._compute_attribute_scores(prefs)

        # Identify preferred and avoided attribute values
        # NOTE: With 3 losers per round, most values end up with low scores
        # Only avoid values with VERY low scores (clearly disliked)
        preferred_attrs = {}  # attr_name -> [values]
        avoided_attrs = {}    # attr_name -> [values]

        for attr_name, scores in attr_scores.items():
            preferred_attrs[attr_name] = []
            avoided_attrs[attr_name] = []

            if not scores:
                continue

            # Calculate mean score to use relative thresholds
            mean_score = sum(scores.values()) / len(scores) if scores else 0.5

            for value, score in scores.items():
                # Preferred: above mean AND above 0.5
                if score >= max(0.5, mean_score + 0.1):
                    preferred_attrs[attr_name].append(value)
                # Avoided: only if VERY low (below 0.25) AND seen enough times
                # This ensures we only filter truly disliked patterns
                elif score <= 0.25:
                    # Check if we have enough data points
                    count = prefs.attribute_counts.get(attr_name, {}).get(value, 0)
                    if count >= 4:  # Only avoid if seen 4+ times
                        avoided_attrs[attr_name].append(value)

        for category in prefs.category_order:
            # Get all items in this category
            category_item_ids = []
            for cluster in self.category_clusters.get(category, set()):
                items = self.cluster_category_items.get(cluster, {}).get(category, [])
                category_item_ids.extend(items)

            # Remove duplicates and seen items
            category_item_ids = [i for i in set(category_item_ids) if i not in seen_ids]

            if not category_item_ids:
                continue

            # Score and filter items
            scored_items = []

            # Get per-category onboarding preferences for soft scoring
            onboarding_cat = self._map_tinder_to_onboarding_category(category)
            cat_onboarding_prefs = prefs.onboarding_prefs.get(onboarding_cat, {})

            for item_id in category_item_ids:
                item_attrs = self.item_attributes.get(item_id, {})
                info = self.get_item_info(item_id)

                # ============================================
                # HARD FILTERS FROM ONBOARDING
                # ============================================

                # Skip avoided colors (onboarding hard filter)
                item_color = (info.get('color', '') or '').lower()
                if prefs.colors_to_avoid and any(c in item_color for c in prefs.colors_to_avoid):
                    continue

                # Skip avoided materials (onboarding hard filter)
                item_fabric = (info.get('fabric', '') or '').lower()
                if prefs.materials_to_avoid and any(m in item_fabric for m in prefs.materials_to_avoid):
                    continue

                # Skip avoided brands (onboarding hard filter)
                item_brand = (info.get('brand', '') or '').lower()
                if prefs.brands_to_avoid and item_brand and item_brand in prefs.brands_to_avoid:
                    continue

                # Skip items with learned avoided attributes (from swipe feedback)
                skip = False
                for attr_name, bad_values in avoided_attrs.items():
                    if item_attrs.get(attr_name) in bad_values:
                        skip = True
                        break
                if skip:
                    continue

                # ============================================
                # SOFT SCORING FROM ONBOARDING
                # ============================================
                onboarding_bonus = 0.0

                # Preferred brand boost (+0.40) - STRONG boost for user's favorite brands
                # This ensures the feed prioritizes items from brands the user loves
                if prefs.preferred_brands and item_brand and item_brand in prefs.preferred_brands:
                    onboarding_bonus += 0.40

                # Per-category preference matching
                if cat_onboarding_prefs:
                    # Fit matching (+0.20)
                    pref_fits = cat_onboarding_prefs.get('fits', [])
                    item_fit = (info.get('fit', '') or '').lower()
                    if pref_fits and item_fit:
                        if item_fit in [f.lower() for f in pref_fits]:
                            onboarding_bonus += 0.20

                    # Sleeve matching (+0.10)
                    pref_sleeves = cat_onboarding_prefs.get('sleeves', [])
                    item_sleeve = (info.get('sleeve', '') or '').lower()
                    if pref_sleeves and item_sleeve:
                        if item_sleeve in [s.lower() for s in pref_sleeves]:
                            onboarding_bonus += 0.10

                    # Neckline matching (+0.15)
                    pref_necklines = cat_onboarding_prefs.get('necklines', [])
                    item_neckline = (info.get('neckline', '') or '').lower()
                    if pref_necklines and item_neckline:
                        if item_neckline in [n.lower() for n in pref_necklines]:
                            onboarding_bonus += 0.15

                # Calculate attribute match score
                attr_match = 0
                attr_count = 0
                for attr_name, good_values in preferred_attrs.items():
                    if good_values:  # Only count if we have preferences
                        attr_count += 1
                        if item_attrs.get(attr_name) in good_values:
                            attr_match += 1

                # Compute overall score
                # Attribute score (from learned preferences)
                item_attr_score = self._get_item_attribute_score(item_id, attr_scores)

                # Taste vector similarity
                taste_sim = 0.5
                emb = self.embeddings_data.get(item_id, {}).get('embedding')
                if emb is not None and prefs.taste_vector is not None:
                    emb_norm = emb / np.linalg.norm(emb)
                    taste_sim = float(np.dot(emb_norm, prefs.taste_vector))
                    taste_sim = (taste_sim + 1) / 2  # Normalize to 0-1

                # Cluster score (legacy)
                cluster = self.item_to_cluster.get(item_id)
                cat_scores = prefs.category_cluster_scores.get(category, {})
                cat_counts = prefs.category_cluster_counts.get(category, {})
                cluster_score = 0.5
                if cluster and cat_counts.get(cluster, 0) > 0:
                    cluster_score = (cat_scores.get(cluster, 0) + 1) / (cat_counts.get(cluster, 0) + 2)

                # Combined score: prioritize onboarding preferences + learned attributes
                # Updated formula with stronger onboarding bonus for brand prioritization
                # onboarding_bonus can be up to 0.85 (brand=0.40, fit=0.20, sleeve=0.10, neckline=0.15)
                combined_score = (
                    0.30 * item_attr_score +      # Learned attribute preferences
                    0.20 * taste_sim +             # Visual taste similarity
                    0.10 * cluster_score +         # Cluster preference (legacy)
                    0.40 * min(onboarding_bonus, 1.0)  # Onboarding preferences (capped)
                )

                scored_items.append({
                    'id': item_id,
                    'category': category,
                    'attr_score': item_attr_score,
                    'taste_sim': taste_sim,
                    'cluster_score': cluster_score,
                    'onboarding_bonus': onboarding_bonus,
                    'combined': combined_score,
                    'attrs': item_attrs
                })

            # Sort by combined score
            scored_items.sort(key=lambda x: x['combined'], reverse=True)

            # Format output
            category_items = []
            # Use gender-aware image path
            img_prefix = "/women-images" if prefs.gender == "female" else "/images"
            for item in scored_items[:items_per_category]:
                info = self.get_item_info(item['id'])
                category_items.append({
                    'id': item['id'],
                    'category': category,
                    'image_url': f"{img_prefix}/{item['id'].replace(' ', '%20')}.webp",
                    'brand': info.get('brand', ''),
                    'color': info.get('color', ''),
                    'similarity': round(item['taste_sim'], 3),
                    'cluster_match': round(item['cluster_score'], 3),
                    'attr_match': round(item['attr_score'], 3),
                    'onboarding_match': round(item.get('onboarding_bonus', 0), 3),
                    'combined': round(item['combined'], 3),
                    # Include detected attributes for debugging
                    'attributes': item['attrs']
                })

            result[category] = category_items

        return result

    def get_learned_style_for_category(
        self,
        prefs: PredictivePreferences,
        category: str
    ) -> Dict:
        """Get the learned style preferences for a specific category."""
        scores = prefs.category_cluster_scores.get(category, {})
        counts = prefs.category_cluster_counts.get(category, {})

        if not counts:
            return {'status': 'not_learned', 'category': category}

        # Find top cluster
        cluster_prefs = []
        for cluster, count in counts.items():
            wins = scores.get(cluster, 0)
            pref_score = (wins + 1) / (count + 2)
            profile = self._get_cluster_profile(cluster)
            cluster_prefs.append({
                'cluster': cluster,
                'score': pref_score,
                'profile': profile
            })

        cluster_prefs.sort(key=lambda x: x['score'], reverse=True)
        top = cluster_prefs[0] if cluster_prefs else None

        return {
            'status': 'learned',
            'category': category,
            'preferred_cluster': top['cluster'] if top else None,
            'preferred_style': top['profile'].get('description', '') if top else '',
            'preferred_archetype': top['profile'].get('dominant_archetype', '') if top else '',
            'confidence': top['score'] if top else 0,
            'all_cluster_prefs': cluster_prefs
        }


# Test
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Predictive Four-Choice Engine Test')
    parser.add_argument(
        '--gender', '-g',
        choices=['male', 'female', 'men', 'women', 'm', 'f'],
        default='male',
        help='Gender for the test session (default: male)'
    )
    parser.add_argument(
        '--max-rounds',
        type=int,
        default=30,
        help='Maximum rounds to simulate (default: 30)'
    )
    args = parser.parse_args()

    gender = args.gender
    if gender in ('men', 'm'):
        gender = 'male'
    elif gender in ('women', 'f'):
        gender = 'female'

    print(f"=== Predictive Four-Choice Engine Test (Gender: {gender}) ===\n")

    try:
        engine = PredictiveFourEngine(gender=gender)
        prefs = PredictivePreferences(user_id="test_user", gender=gender)
        prefs = engine.initialize_session(prefs)

        print(f"Gender: {prefs.gender}")
        print(f"Categories to test: {prefs.category_order}")
        print()

        # Simulate session
        for round_num in range(args.max_rounds):
            items, test_info = engine.get_four_items(prefs)

            if not items:
                print(f"\n=== All categories complete after {round_num} rounds ===")
                break

            category = test_info['category']
            prediction = test_info['prediction']

            print(f"Round {round_num + 1} - {category}")
            print(f"  Clusters shown: {test_info['clusters_shown']}")
            print(f"  Prediction: cluster {prediction.get('predicted_cluster')} (conf: {prediction.get('confidence', 0):.3f})")

            # Simulate user choice (prefer lower-numbered clusters for testing)
            winner_idx = 0  # Always pick first item
            winner = items[winner_idx]

            prefs, result = engine.record_choice(prefs, winner)

            print(f"  User picked: cluster {result['winner_cluster']}")
            print(f"  Prediction correct: {result['prediction_correct']}")
            print(f"  Consecutive correct: {result['consecutive_correct']}")

            if result['category_complete']:
                print(f"  >>> Category '{category}' COMPLETE!")
                if result.get('all_complete'):
                    print(f"\n=== ALL CATEGORIES COMPLETE ===")
                    break

            print()

        # Summary
        summary = engine.get_session_summary(prefs)
        print(f"\n=== Session Summary ===")
        print(f"Gender: {summary.get('gender', 'unknown')}")
        print(f"Total rounds: {summary['rounds_completed']}")
        print(f"Prediction accuracy: {summary['prediction_accuracy']*100:.1f}%")
        print(f"Categories completed: {summary['categories_completed']}")

        print(f"\nPer-category stats:")
        for cat, stats in summary['category_stats'].items():
            status = "COMPLETE" if stats['complete'] else "incomplete"
            print(f"  {cat}: {stats['rounds']} rounds [{status}]")
            for cp in stats['cluster_preferences']:
                profile_str = cp.get('profile', '')[:40]
                print(f"    Cluster {cp['cluster']}: {cp['score']:.2f} ({cp['wins']}/{cp['total']}) - {profile_str}")

    except Exception as e:
        print(f"Error: {e}")
        print(f"\nNote: For gender='female', you need to have women's embeddings generated.")
        print(f"Run: python3 women_embeddings.py")
        import traceback
        traceback.print_exc()
