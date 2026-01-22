"""
Swipe-based Style Learning Engine v4 (Feedback-Driven)

KEY INSIGHT: Separate EXPLORATION from INTERPRETATION

EXPLORATION (Selection):
- Use CLIP embeddings for visual similarity
- Use K-means clusters for coverage (visual regions, not semantic)
- RESPECT NEGATIVE FEEDBACK: consecutive dislikes = move away
- Adaptive: reduce exploration in regions with strong negative signal

INTERPRETATION (After learning):
- Use taxonomy to EXPLAIN what user likes
- Archetypes, anchors, attributes for human understanding
- Brand tracked separately

This works because:
- Selection is VISUAL (embeddings) - accurate
- Interpretation is SEMANTIC (taxonomy) - interpretable
- Feedback is RESPECTED - good UX
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import random
from collections import Counter
from sklearn.cluster import KMeans


class SwipeAction(Enum):
    LIKE = 1
    DISLIKE = -1
    SKIP = 0


@dataclass
class UserPreferences:
    """User's learned preferences from swipes."""
    user_id: str

    # Hard filters (pre-set)
    colors_to_avoid: Set[str] = field(default_factory=set)
    materials_to_avoid: Set[str] = field(default_factory=set)  # Pre-filter: avoid these fabrics
    selected_categories: Set[str] = field(default_factory=set)  # Pre-filter: only show these categories
    selected_necklines: Set[str] = field(default_factory=set)  # Pre-filter: only show these necklines
    selected_sleeves: Set[str] = field(default_factory=set)  # Pre-filter: only show these sleeve lengths

    # Brand filtering (from onboarding)
    brands_to_avoid: Set[str] = field(default_factory=set)  # Hard filter: exclude these brands
    preferred_brands: Set[str] = field(default_factory=set)  # Soft boost: prefer these brands

    # Per-category onboarding preferences (for soft scoring in feed)
    onboarding_prefs: Dict = field(default_factory=dict)  # category -> {fits, sleeves, necklines, types}

    # Learned from swipes
    liked_ids: List[str] = field(default_factory=list)
    disliked_ids: List[str] = field(default_factory=list)
    skipped_ids: List[str] = field(default_factory=list)

    # Recent feedback for anti-repetition
    recent_actions: List[Tuple[str, SwipeAction]] = field(default_factory=list)

    # Taste vector (CLIP embedding space)
    taste_vector: Optional[np.ndarray] = None
    anti_taste_vector: Optional[np.ndarray] = None  # What user DOESN'T like

    # Cluster tracking (for visual coverage)
    swipes_per_cluster: Dict[int, int] = field(default_factory=dict)
    likes_per_cluster: Dict[int, int] = field(default_factory=dict)
    dislikes_per_cluster: Dict[int, int] = field(default_factory=dict)

    # Region rejection tracking
    rejected_regions: Set[int] = field(default_factory=set)  # Clusters with strong negative signal
    rejected_categories: Set[str] = field(default_factory=set)  # Categories with strong negative signal

    # Brand tracking (separate)
    brand_likes: Dict[str, int] = field(default_factory=dict)
    brand_dislikes: Dict[str, int] = field(default_factory=dict)

    # Attribute tracking
    attribute_likes: Dict[str, Dict[str, int]] = field(default_factory=dict)
    attribute_dislikes: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Taste stability tracking
    taste_vector_history: List[np.ndarray] = field(default_factory=list)

    # Session state
    exploration_complete: bool = False

    @property
    def total_swipes(self) -> int:
        return len(self.liked_ids) + len(self.disliked_ids)

    @property
    def seen_ids(self) -> Set[str]:
        return set(self.liked_ids + self.disliked_ids + self.skipped_ids)

    def get_consecutive_dislikes(self) -> int:
        """Count consecutive recent dislikes."""
        count = 0
        for item_id, action in reversed(self.recent_actions):
            if action == SwipeAction.DISLIKE:
                count += 1
            else:
                break
        return count


class SwipeEngine:
    """
    Feedback-driven swipe engine.

    Key principles:
    1. VISUAL clusters for exploration (not semantic archetypes)
    2. RESPECT consecutive dislikes - move away fast
    3. ADAPTIVE - reject regions with strong negative signal
    4. INTERPRET with taxonomy AFTER learning
    """

    # Visual clustering (for exploration coverage)
    N_VISUAL_CLUSTERS = 12  # Visual regions, not semantic categories

    # Exploration settings
    MIN_SWIPES_PER_CLUSTER = 2  # Soft target, not forced
    MAX_CONSECUTIVE_SAME = 2   # Don't show same cluster twice in a row

    # FEEDBACK SETTINGS (crucial for UX)
    CONSECUTIVE_DISLIKE_THRESHOLD = 2  # After 2 dislikes in a row, switch strategy
    CLUSTER_REJECTION_RATIO = 0.80     # If 80%+ dislikes in cluster, mark as rejected
    MIN_CLUSTER_SAMPLES_FOR_REJECTION = 2  # Need 2+ samples to reject (lowered!)

    # NOTE: Removed category-based rejection - too coarse-grained
    # Instead, we learn WITHIN categories using visual/taxonomy signals

    # Stopping
    MIN_TOTAL_SWIPES = 40  # Increased for better learning
    TASTE_STABILITY_THRESHOLD = 0.95

    def __init__(
        self,
        embeddings_path: str = "/home/ubuntu/recSys/outfitTransformer/models/hp_embeddings.pkl",
        taxonomy_path: str = "/home/ubuntu/recSys/outfitTransformer/models/hp_taxonomy_scores.pkl",
        csv_dir: str = "/home/ubuntu/recSys/outfitTransformer/HPdataset"
    ):
        # Load embeddings
        with open(embeddings_path, 'rb') as f:
            self.embeddings_data = pickle.load(f)

        # Load taxonomy (for interpretation only)
        try:
            with open(taxonomy_path, 'rb') as f:
                self.taxonomy = pickle.load(f)
        except:
            self.taxonomy = None
            print("Warning: No taxonomy scores found, interpretation will be limited")

        # Load CSV metadata
        self.metadata = self._load_metadata(csv_dir)

        # Load duplicate exclusions
        self.excluded_items = set()
        try:
            import json
            exclusion_path = Path(csv_dir) / "duplicate_exclusions.json"
            if exclusion_path.exists():
                with open(exclusion_path, 'r') as f:
                    self.excluded_items = set(json.load(f))
                print(f"  Excluding {len(self.excluded_items)} duplicate items")
        except Exception as e:
            print(f"Warning: Could not load duplicate exclusions: {e}")

        # Create lookup structures (excluding duplicates)
        self.item_ids = [k for k in self.embeddings_data.keys() if k not in self.excluded_items]
        self.embeddings_matrix = np.array([
            self.embeddings_data[k]['embedding'] for k in self.item_ids
        ])

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)
        self.embeddings_matrix_normed = self.embeddings_matrix / norms

        # Create VISUAL clusters (for exploration coverage)
        self._create_visual_clusters()

        # Get unique brands and attributes
        self._index_metadata()

        print(f"SwipeEngine loaded: {len(self.item_ids)} items")
        print(f"  Visual clusters: {self.N_VISUAL_CLUSTERS}")
        cluster_sizes = [len(self.cluster_items[c]) for c in range(self.N_VISUAL_CLUSTERS)]
        print(f"  Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={np.mean(cluster_sizes):.0f}")
        print(f"  Unique brands: {len(self.all_brands)}")

    def _create_visual_clusters(self):
        """Create clusters from VISUAL embeddings (not semantic)."""
        kmeans = KMeans(n_clusters=self.N_VISUAL_CLUSTERS, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(self.embeddings_matrix_normed)
        self.cluster_centers = kmeans.cluster_centers_

        # Map items to clusters
        self.item_to_cluster = {
            item_id: label
            for item_id, label in zip(self.item_ids, self.cluster_labels)
        }

        # Items per cluster
        self.cluster_items = {c: [] for c in range(self.N_VISUAL_CLUSTERS)}
        for item_id, cluster in self.item_to_cluster.items():
            self.cluster_items[cluster].append(item_id)

    def _index_metadata(self):
        """Index metadata for quick lookup."""
        self.all_brands = set()
        self.attribute_values = {
            'color': set(),
            'fit': set(),
            'fabric': set(),
            'category': set()
        }
        for meta in self.metadata.values():
            if meta.get('brand'):
                self.all_brands.add(meta['brand'])
            for attr in self.attribute_values.keys():
                val = meta.get(attr, '')
                if val:
                    self.attribute_values[attr].add(val)

    def _load_metadata(self, csv_dir: str) -> Dict[str, Dict]:
        """Load item metadata from CSVs."""
        csv_path = Path(csv_dir)
        metadata = {}

        # Plain T-shirts
        try:
            plain = pd.read_csv(csv_path / "Formatted - Plain T-shirts.csv")
            for _, row in plain.iterrows():
                item_id = f"Plain T-shirts/{int(row['Item List'])}"
                metadata[item_id] = {
                    'color': str(row['Color ']).strip().lower(),
                    'fit': str(row['Fit']).strip(),
                    'brand': str(row['Brand ']).strip(),
                    'fabric': str(row['Fabric']).strip().lower(),
                    'neckline': str(row['Neckline']).strip().lower(),
                    'sleeve': str(row['Sleeve']).strip().lower(),
                    'category': 'Plain T-shirts'
                }
        except Exception as e:
            print(f"Warning: Could not load Plain T-shirts: {e}")

        # Graphics T-shirts
        try:
            graphics = pd.read_csv(csv_path / "Formatted - Graphics T-shirts.csv")
            for _, row in graphics.iterrows():
                item_id = f"Graphics T-shirts/{int(row['Image title'])}"
                metadata[item_id] = {
                    'color': str(row['Color ']).strip().lower(),
                    'fit': str(row['Fit']).strip(),
                    'brand': str(row['Brand ']).strip(),
                    'fabric': str(row['Fabric']).strip().lower(),
                    'neckline': str(row['Neckline']).strip().lower() if pd.notna(row.get('Neckline')) else '',
                    'sleeve': str(row['Sleeve']).strip().lower() if pd.notna(row.get('Sleeve')) else '',
                    'category': 'Graphics T-shirts'
                }
        except Exception as e:
            print(f"Warning: Could not load Graphics T-shirts: {e}")

        # Small logos
        try:
            small = pd.read_csv(csv_path / "Formatted - Small logos.csv")
            for _, row in small.iterrows():
                if pd.isna(row['Item List']):
                    continue
                item_id = f"Small logos/{int(row['Item List'])}"
                metadata[item_id] = {
                    'color': str(row.get('Color ', '')).strip().lower() if pd.notna(row.get('Color ')) else '',
                    'fit': str(row.get('Fit', '')).strip() if pd.notna(row.get('Fit')) else '',
                    'brand': str(row.get('Brand ', '')).strip() if pd.notna(row.get('Brand ')) else '',
                    'fabric': str(row.get('Fabric', '')).strip().lower() if pd.notna(row.get('Fabric')) else '',
                    'neckline': str(row.get('Neckline', '')).strip().lower() if pd.notna(row.get('Neckline')) else '',
                    'sleeve': str(row.get('Sleeve', '')).strip().lower() if pd.notna(row.get('Sleeve')) else '',
                    'category': 'Small logos'
                }
        except Exception as e:
            print(f"Warning: Could not load Small logos: {e}")

        # Athletic T-shirts (uses row number as item ID, 1-indexed)
        try:
            athletic = pd.read_csv(csv_path / "Formatted - Athletic T-shirts.csv")
            for idx, row in athletic.iterrows():
                item_id = f"Athletic/{idx + 1}"  # 1-indexed
                metadata[item_id] = {
                    'color': str(row.get('Color ', '')).strip().lower() if pd.notna(row.get('Color ')) else '',
                    'fit': str(row.get('Fit', '')).strip() if pd.notna(row.get('Fit')) else '',
                    'brand': str(row.get('Brand ', '')).strip() if pd.notna(row.get('Brand ')) else '',
                    'fabric': str(row.get('Fabric', '')).strip().lower() if pd.notna(row.get('Fabric')) else '',
                    'neckline': str(row.get('Neckline', '')).strip().lower() if pd.notna(row.get('Neckline')) else '',
                    'sleeve': str(row.get('Sleeve', '')).strip().lower() if pd.notna(row.get('Sleeve')) else '',
                    'category': 'Athletic'
                }
        except Exception as e:
            print(f"Warning: Could not load Athletic T-shirts: {e}")

        return metadata

    def get_candidates(
        self,
        prefs: UserPreferences,
        exclude_seen: bool = True
    ) -> List[str]:
        """Get candidate items after applying filters."""
        candidates = []

        for item_id in self.item_ids:
            if exclude_seen and item_id in prefs.seen_ids:
                continue

            meta = self.metadata.get(item_id, {})
            item_color = meta.get('color', '').lower()
            item_category = meta.get('category', '')
            item_neckline = meta.get('neckline', '').lower()
            item_fabric = meta.get('fabric', '').lower()
            item_sleeve = meta.get('sleeve', '').lower()

            # Filter: colors to avoid (user preference - hard filter)
            if any(avoid in item_color for avoid in prefs.colors_to_avoid):
                continue

            # Filter: materials to avoid (user preference - hard filter)
            if any(avoid in item_fabric for avoid in prefs.materials_to_avoid):
                continue

            # Filter: brands to avoid (user preference - hard filter)
            item_brand = meta.get('brand', '').lower()
            if prefs.brands_to_avoid and item_brand and item_brand in prefs.brands_to_avoid:
                continue

            # Filter: only show selected categories (pre-filter from setup)
            # If selected_categories is empty, show all categories (no filter)
            if prefs.selected_categories and item_category not in prefs.selected_categories:
                continue

            # Filter: only show selected necklines (pre-filter from setup)
            # If selected_necklines is empty, show all necklines (no filter)
            if prefs.selected_necklines and item_neckline not in prefs.selected_necklines:
                continue

            # Filter: only show selected sleeve lengths (pre-filter from setup)
            # If selected_sleeves is empty, show all sleeves (no filter)
            if prefs.selected_sleeves and item_sleeve not in prefs.selected_sleeves:
                continue

            candidates.append(item_id)

        return candidates

    def _get_cluster_health(self, prefs: UserPreferences, cluster: int) -> float:
        """
        Get health score for a cluster (1.0 = good, 0.0 = rejected).

        Based on like/dislike ratio with Bayesian smoothing.
        """
        likes = prefs.likes_per_cluster.get(cluster, 0)
        dislikes = prefs.dislikes_per_cluster.get(cluster, 0)
        total = likes + dislikes

        if total < self.MIN_CLUSTER_SAMPLES_FOR_REJECTION:
            return 0.5  # Not enough data, neutral

        # Bayesian smoothing
        like_rate = (likes + 1) / (total + 2)
        return like_rate

    def _should_reject_cluster(self, prefs: UserPreferences, cluster: int) -> bool:
        """Check if cluster should be rejected due to strong negative signal."""
        likes = prefs.likes_per_cluster.get(cluster, 0)
        dislikes = prefs.dislikes_per_cluster.get(cluster, 0)
        total = likes + dislikes

        if total < self.MIN_CLUSTER_SAMPLES_FOR_REJECTION:
            return False

        dislike_rate = dislikes / total
        return dislike_rate >= self.CLUSTER_REJECTION_RATIO

    def _get_item_taxonomy(self, item_id: str) -> Optional[Dict]:
        """Get taxonomy scores for an item (archetypes + anchors)."""
        if not self.taxonomy:
            return None
        return self.taxonomy.get('item_scores', {}).get(item_id)

    def _compute_taxonomy_preferences(self, prefs: UserPreferences) -> Dict:
        """Compute archetype and anchor preferences from liked/disliked items.

        This gives semantic meaning to user preferences beyond just categories.
        """
        if not self.taxonomy:
            return {}

        archetype_names = self.taxonomy.get('archetype_names', [])
        anchor_names = self.taxonomy.get('anchor_names', [])

        # Accumulate scores from liked items
        liked_arch = {a: [] for a in archetype_names}
        liked_anchor = {a: [] for a in anchor_names}

        for item_id in prefs.liked_ids:
            tax = self._get_item_taxonomy(item_id)
            if tax:
                for arch, score in tax.get('archetype_scores', {}).items():
                    liked_arch[arch].append(score)
                for anchor, score in tax.get('anchor_scores', {}).items():
                    liked_anchor[anchor].append(score)

        # Accumulate scores from disliked items
        disliked_arch = {a: [] for a in archetype_names}
        disliked_anchor = {a: [] for a in anchor_names}

        for item_id in prefs.disliked_ids:
            tax = self._get_item_taxonomy(item_id)
            if tax:
                for arch, score in tax.get('archetype_scores', {}).items():
                    disliked_arch[arch].append(score)
                for anchor, score in tax.get('anchor_scores', {}).items():
                    disliked_anchor[anchor].append(score)

        # Compute preference scores (liked_avg - disliked_avg)
        archetype_prefs = {}
        for arch in archetype_names:
            liked_avg = np.mean(liked_arch[arch]) if liked_arch[arch] else 0.5
            disliked_avg = np.mean(disliked_arch[arch]) if disliked_arch[arch] else 0.5
            archetype_prefs[arch] = {
                'liked_avg': round(liked_avg, 3),
                'disliked_avg': round(disliked_avg, 3),
                'preference': round(liked_avg - disliked_avg, 3)  # Positive = likes this archetype
            }

        anchor_prefs = {}
        for anchor in anchor_names:
            liked_avg = np.mean(liked_anchor[anchor]) if liked_anchor[anchor] else 0.5
            disliked_avg = np.mean(disliked_anchor[anchor]) if disliked_anchor[anchor] else 0.5
            anchor_prefs[anchor] = {
                'liked_avg': round(liked_avg, 3),
                'disliked_avg': round(disliked_avg, 3),
                'preference': round(liked_avg - disliked_avg, 3)
            }

        return {
            'archetypes': archetype_prefs,
            'anchors': anchor_prefs
        }

    def _get_cluster_profile(self, cluster: int) -> Dict:
        """Get semantic profile of a cluster using taxonomy scores."""
        if not self.taxonomy:
            return {'description': f'Visual cluster {cluster}'}

        # Get items in this cluster
        cluster_items = self.cluster_items.get(cluster, [])
        if not cluster_items:
            return {'description': f'Empty cluster {cluster}'}

        # Aggregate taxonomy scores
        arch_scores = {}
        anchor_scores = {}

        for item_id in cluster_items[:20]:  # Sample first 20 items
            tax = self._get_item_taxonomy(item_id)
            if tax:
                for arch, score in tax.get('archetype_scores', {}).items():
                    if arch not in arch_scores:
                        arch_scores[arch] = []
                    arch_scores[arch].append(score)
                for anchor, score in tax.get('anchor_scores', {}).items():
                    if anchor not in anchor_scores:
                        anchor_scores[anchor] = []
                    anchor_scores[anchor].append(score)

        # Find dominant archetype
        dominant_arch = max(arch_scores.items(), key=lambda x: np.mean(x[1]))[0] if arch_scores else 'unknown'

        # Find top 2 anchors
        anchor_avgs = [(a, np.mean(s)) for a, s in anchor_scores.items()]
        anchor_avgs.sort(key=lambda x: x[1], reverse=True)
        top_anchors = [a[0] for a in anchor_avgs[:2]]

        return {
            'dominant_archetype': dominant_arch,
            'top_anchors': top_anchors,
            'description': f"{dominant_arch} / {', '.join(top_anchors)}"
        }

    def get_next_item(
        self,
        prefs: UserPreferences
    ) -> Optional[Tuple[str, bool]]:
        """
        Get next item with FEEDBACK-DRIVEN selection.

        Strategy:
        1. Check for consecutive dislikes - if many, switch to exploitation
        2. Avoid rejected clusters
        3. Explore underexplored non-rejected clusters
        4. Use visual diversity within valid clusters
        """
        candidates = self.get_candidates(prefs)

        if not candidates:
            return None, True

        # Note: Rejected regions and categories are now updated in record_swipe()
        # so they take effect immediately after feedback

        # Check stopping condition
        non_rejected_clusters = [c for c in range(self.N_VISUAL_CLUSTERS) if c not in prefs.rejected_regions]

        if prefs.total_swipes >= self.MIN_TOTAL_SWIPES:
            # Check if we've explored enough non-rejected clusters
            explored_non_rejected = sum(
                1 for c in non_rejected_clusters
                if prefs.swipes_per_cluster.get(c, 0) >= self.MIN_SWIPES_PER_CLUSTER
            )

            if explored_non_rejected >= len(non_rejected_clusters) * 0.7:  # 70% coverage of non-rejected
                # Check taste stability
                if len(prefs.taste_vector_history) >= 3:
                    recent = prefs.taste_vector_history[-3:]
                    similarities = [np.dot(recent[i], recent[i+1]) for i in range(len(recent)-1)]
                    if np.mean(similarities) >= self.TASTE_STABILITY_THRESHOLD:
                        prefs.exploration_complete = True
                        return None, True

        # FEEDBACK-DRIVEN SELECTION

        consecutive_dislikes = prefs.get_consecutive_dislikes()

        # If many consecutive dislikes, IMMEDIATELY switch to taste-based
        if consecutive_dislikes >= self.CONSECUTIVE_DISLIKE_THRESHOLD and prefs.taste_vector is not None:
            # Find items AWAY from anti-taste and CLOSE to taste
            return self._select_recovery_item(prefs, candidates), False

        # Filter out rejected clusters
        valid_candidates = [
            c for c in candidates
            if self.item_to_cluster.get(c) not in prefs.rejected_regions
        ]

        if not valid_candidates:
            valid_candidates = candidates  # Fallback

        # Check for cooldown (don't repeat same cluster)
        last_cluster = None
        if prefs.recent_actions:
            last_item = prefs.recent_actions[-1][0]
            last_cluster = self.item_to_cluster.get(last_item)

        # Find underexplored clusters (excluding rejected and last shown)
        underexplored = [
            c for c in non_rejected_clusters
            if prefs.swipes_per_cluster.get(c, 0) < self.MIN_SWIPES_PER_CLUSTER
            and c != last_cluster
        ]

        if underexplored:
            # RANDOMIZED cluster selection with health weighting
            # Pick randomly among underexplored clusters, weighted by health
            cluster_health = [(c, self._get_cluster_health(prefs, c)) for c in underexplored]

            # Use softmax-style weighting for randomness
            healths = np.array([h for _, h in cluster_health])
            weights = np.exp(healths * 2)  # Temperature=0.5 for moderate randomness
            weights = weights / weights.sum()

            # Randomly select cluster based on weights
            selected_idx = np.random.choice(len(cluster_health), p=weights)
            target_cluster = cluster_health[selected_idx][0]

            cluster_candidates = [
                c for c in valid_candidates
                if self.item_to_cluster.get(c) == target_cluster
            ]
            if cluster_candidates:
                # Random selection within cluster (not just diverse)
                return random.choice(cluster_candidates), False

        # INCREASED RANDOMNESS in exploitation
        # 50% random, 50% taste-based (was 30/70)
        if prefs.taste_vector is not None and random.random() > 0.5:
            return self._select_taste_item(prefs, valid_candidates), False
        else:
            # Random from valid non-last-cluster candidates
            diverse_candidates = [
                c for c in valid_candidates
                if self.item_to_cluster.get(c) != last_cluster
            ]
            if not diverse_candidates:
                diverse_candidates = valid_candidates

            return random.choice(diverse_candidates), False

    def _select_recovery_item(self, prefs: UserPreferences, candidates: List[str]) -> str:
        """Select item to recover from consecutive dislikes with diversity."""
        # Find items closest to taste vector AND far from recent dislikes
        candidate_embeddings = np.array([
            self.embeddings_data[c]['embedding'] for c in candidates
        ])
        candidate_embeddings_normed = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)

        # Taste similarity
        taste_scores = candidate_embeddings_normed @ prefs.taste_vector

        # Penalty for similarity to recent dislikes
        if prefs.anti_taste_vector is not None:
            anti_scores = candidate_embeddings_normed @ prefs.anti_taste_vector
            # Combined score: high taste, low anti-taste
            combined = taste_scores - 0.5 * anti_scores
        else:
            combined = taste_scores

        # Get recent attributes for diversity
        recent_attrs = self._get_recent_attributes(prefs, n=3)

        # Pick from top candidates with diversity filter
        top_k = min(15, len(candidates))  # Expanded pool
        top_indices = np.argsort(combined)[-top_k:]

        # Filter for diversity
        diverse_indices = [
            idx for idx in top_indices
            if self._is_diverse_enough(candidates[idx], recent_attrs, strict=True)
        ]

        if not diverse_indices:
            diverse_indices = list(top_indices[:5])  # Fallback to top 5

        return candidates[random.choice(diverse_indices)]

    def _get_recent_attributes(self, prefs: UserPreferences, n: int = 3) -> List[Dict]:
        """Get attributes of last n shown items."""
        recent_attrs = []
        for item_id, _ in reversed(prefs.recent_actions[-n:]):
            meta = self.metadata.get(item_id, {})
            recent_attrs.append({
                'color': meta.get('color', ''),
                'category': meta.get('category', ''),
                'brand': meta.get('brand', '')
            })
        return recent_attrs

    def _is_diverse_enough(self, item_id: str, recent_attrs: List[Dict], strict: bool = True) -> bool:
        """Check if item is different enough from recent items.

        Strict mode: Must differ in color OR category from ALL recent items
        Relaxed mode: Must differ in color OR category from at least ONE recent item
        """
        if not recent_attrs:
            return True

        meta = self.metadata.get(item_id, {})
        item_color = meta.get('color', '')
        item_category = meta.get('category', '')

        if strict:
            # Must be different from ALL recent items in at least one attribute
            for recent in recent_attrs:
                same_color = item_color and item_color == recent['color']
                same_category = item_category and item_category == recent['category']
                # If same color AND same category as ANY recent item, not diverse
                if same_color and same_category:
                    return False
            return True
        else:
            # Relaxed: just different from the most recent item
            if recent_attrs:
                most_recent = recent_attrs[0]
                same_color = item_color and item_color == most_recent['color']
                same_category = item_category and item_category == most_recent['category']
                return not (same_color and same_category)
            return True

    def _select_taste_item(self, prefs: UserPreferences, candidates: List[str]) -> str:
        """Select item based on taste vector with DIVERSITY ENFORCEMENT."""
        candidate_embeddings = np.array([
            self.embeddings_data[c]['embedding'] for c in candidates
        ])
        candidate_embeddings_normed = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)

        similarities = candidate_embeddings_normed @ prefs.taste_vector

        # Get recent item attributes for diversity check
        recent_attrs = self._get_recent_attributes(prefs, n=3)

        # Pick from top candidates (expanded pool for diversity filtering)
        top_k = min(30, len(candidates))  # Expanded from 8 to 30
        top_indices = np.argsort(similarities)[-top_k:]

        # Filter for diversity - must differ from recent items
        diverse_indices = [
            idx for idx in top_indices
            if self._is_diverse_enough(candidates[idx], recent_attrs, strict=True)
        ]

        # Fallback: relax diversity constraint
        if not diverse_indices:
            diverse_indices = [
                idx for idx in top_indices
                if self._is_diverse_enough(candidates[idx], recent_attrs, strict=False)
            ]

        # Final fallback: use all top candidates
        if not diverse_indices:
            diverse_indices = list(top_indices)

        # Weight by similarity for soft selection
        diverse_sims = similarities[diverse_indices]
        weights = np.exp(diverse_sims * 2)  # Lower temperature for more randomness
        weights = weights / weights.sum()

        selected_idx = np.random.choice(len(diverse_indices), p=weights)
        return candidates[diverse_indices[selected_idx]]

    def _select_diverse_item(self, prefs: UserPreferences, candidates: List[str]) -> str:
        """Select item that's different from recently shown."""
        if len(candidates) == 1:
            return candidates[0]

        if not prefs.recent_actions:
            return random.choice(candidates)

        # Get embeddings of recent items
        recent_ids = [item_id for item_id, _ in prefs.recent_actions[-5:]]
        recent_embeddings = []
        for item_id in recent_ids:
            if item_id in self.embeddings_data:
                emb = self.embeddings_data[item_id]['embedding']
                recent_embeddings.append(emb / np.linalg.norm(emb))

        if not recent_embeddings:
            return random.choice(candidates)

        recent_mean = np.mean(recent_embeddings, axis=0)
        recent_mean = recent_mean / np.linalg.norm(recent_mean)

        # Find candidates most DIFFERENT from recent
        candidate_embeddings = np.array([
            self.embeddings_data[c]['embedding'] for c in candidates
        ])
        candidate_embeddings_normed = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)

        similarities = candidate_embeddings_normed @ recent_mean

        # Pick from bottom 3 (most different)
        bottom_k = min(3, len(candidates))
        bottom_indices = np.argsort(similarities)[:bottom_k]

        return candidates[random.choice(bottom_indices)]

    def record_swipe(
        self,
        prefs: UserPreferences,
        item_id: str,
        action: SwipeAction
    ) -> UserPreferences:
        """Record swipe and update all tracking."""
        meta = self.metadata.get(item_id, {})
        cluster = self.item_to_cluster.get(item_id)

        # Track recent actions
        prefs.recent_actions.append((item_id, action))
        if len(prefs.recent_actions) > 10:
            prefs.recent_actions = prefs.recent_actions[-10:]

        is_like = action == SwipeAction.LIKE

        if action == SwipeAction.LIKE:
            prefs.liked_ids.append(item_id)
        elif action == SwipeAction.DISLIKE:
            prefs.disliked_ids.append(item_id)
        else:
            prefs.skipped_ids.append(item_id)
            return prefs

        # Update cluster tracking
        if cluster is not None:
            prefs.swipes_per_cluster[cluster] = prefs.swipes_per_cluster.get(cluster, 0) + 1
            if is_like:
                prefs.likes_per_cluster[cluster] = prefs.likes_per_cluster.get(cluster, 0) + 1
            else:
                prefs.dislikes_per_cluster[cluster] = prefs.dislikes_per_cluster.get(cluster, 0) + 1

        # Update brand tracking
        brand = meta.get('brand', '')
        if brand:
            if is_like:
                prefs.brand_likes[brand] = prefs.brand_likes.get(brand, 0) + 1
            else:
                prefs.brand_dislikes[brand] = prefs.brand_dislikes.get(brand, 0) + 1

        # Update attribute tracking
        for attr_type in ['color', 'fit', 'fabric', 'category']:
            attr_value = meta.get(attr_type, '')
            if attr_value:
                if is_like:
                    if attr_type not in prefs.attribute_likes:
                        prefs.attribute_likes[attr_type] = {}
                    prefs.attribute_likes[attr_type][attr_value] = prefs.attribute_likes[attr_type].get(attr_value, 0) + 1
                else:
                    if attr_type not in prefs.attribute_dislikes:
                        prefs.attribute_dislikes[attr_type] = {}
                    prefs.attribute_dislikes[attr_type][attr_value] = prefs.attribute_dislikes[attr_type].get(attr_value, 0) + 1

        # Update taste vectors
        prefs.taste_vector = self._compute_taste_vector(prefs)
        prefs.anti_taste_vector = self._compute_anti_taste_vector(prefs)

        if prefs.taste_vector is not None:
            prefs.taste_vector_history.append(prefs.taste_vector.copy())
            if len(prefs.taste_vector_history) > 5:
                prefs.taste_vector_history = prefs.taste_vector_history[-5:]

        # Update rejected clusters (visual regions with strong negative signal)
        if cluster is not None and self._should_reject_cluster(prefs, cluster):
            prefs.rejected_regions.add(cluster)

        return prefs

    def _compute_taste_vector(self, prefs: UserPreferences) -> np.ndarray:
        """Compute taste vector from likes."""
        if not prefs.liked_ids:
            return None

        liked_emb = []
        for item_id in prefs.liked_ids:
            if item_id in self.embeddings_data:
                emb = self.embeddings_data[item_id]['embedding']
                liked_emb.append(emb / np.linalg.norm(emb))

        if not liked_emb:
            return None

        taste = np.mean(liked_emb, axis=0)
        taste = taste / np.linalg.norm(taste)

        return taste

    def _compute_anti_taste_vector(self, prefs: UserPreferences) -> np.ndarray:
        """Compute anti-taste vector from recent dislikes (weighted towards recent)."""
        # Weight recent dislikes more heavily
        recent_dislikes = [item_id for item_id, action in prefs.recent_actions[-5:]
                          if action == SwipeAction.DISLIKE]

        if not recent_dislikes:
            return None

        disliked_emb = []
        for item_id in recent_dislikes:
            if item_id in self.embeddings_data:
                emb = self.embeddings_data[item_id]['embedding']
                disliked_emb.append(emb / np.linalg.norm(emb))

        if not disliked_emb:
            return None

        anti_taste = np.mean(disliked_emb, axis=0)
        anti_taste = anti_taste / np.linalg.norm(anti_taste)

        return anti_taste

    def get_item_info(self, item_id: str) -> Dict:
        """Get full item information."""
        meta = self.metadata.get(item_id, {})
        cluster = self.item_to_cluster.get(item_id)

        # Get taxonomy info if available
        dominant_archetype = 'unknown'
        if self.taxonomy and item_id in self.taxonomy.get('item_scores', {}):
            scores = self.taxonomy['item_scores'][item_id]
            dominant_archetype = scores.get('dominant_archetype', 'unknown')

        return {
            'item_id': item_id,
            'image_path': self.embeddings_data.get(item_id, {}).get('path'),
            'cluster': cluster,
            'archetype': dominant_archetype,
            **meta
        }

    def get_preference_summary(self, prefs: UserPreferences) -> Dict:
        """Get comprehensive preference summary with semantic insights."""

        # Cluster health WITH semantic profiles
        cluster_health = {}
        for c in range(self.N_VISUAL_CLUSTERS):
            likes = prefs.likes_per_cluster.get(c, 0)
            dislikes = prefs.dislikes_per_cluster.get(c, 0)
            total = likes + dislikes
            if total > 0:
                profile = self._get_cluster_profile(c)
                cluster_health[c] = {
                    'likes': likes,
                    'dislikes': dislikes,
                    'health': round((likes + 1) / (total + 2), 2),
                    'rejected': c in prefs.rejected_regions,
                    'profile': profile.get('description', f'cluster {c}')
                }

        # Brand preferences
        brand_prefs = {}
        for brand in self.all_brands:
            likes = prefs.brand_likes.get(brand, 0)
            dislikes = prefs.brand_dislikes.get(brand, 0)
            total = likes + dislikes
            if total > 0:
                brand_prefs[brand] = {
                    'score': round((likes + 1) / (total + 2), 2),
                    'likes': likes,
                    'total': total
                }
        sorted_brands = sorted(brand_prefs.items(), key=lambda x: x[1]['score'], reverse=True)
        top_brands = dict(sorted_brands[:5])

        # Attribute preferences
        attribute_prefs = {}
        for attr_type in ['color', 'fit', 'fabric', 'category']:
            likes_dict = prefs.attribute_likes.get(attr_type, {})
            dislikes_dict = prefs.attribute_dislikes.get(attr_type, {})

            all_values = set(likes_dict.keys()) | set(dislikes_dict.keys())
            scored = []
            for val in all_values:
                likes = likes_dict.get(val, 0)
                dislikes = dislikes_dict.get(val, 0)
                total = likes + dislikes
                if total > 0:
                    scored.append({
                        'value': val,
                        'score': round((likes + 1) / (total + 2), 2),
                        'likes': likes,
                        'total': total
                    })

            scored.sort(key=lambda x: x['score'], reverse=True)
            attribute_prefs[attr_type] = scored

        # Coverage stats
        non_rejected = [c for c in range(self.N_VISUAL_CLUSTERS) if c not in prefs.rejected_regions]
        explored = sum(1 for c in non_rejected if prefs.swipes_per_cluster.get(c, 0) >= self.MIN_SWIPES_PER_CLUSTER)
        coverage_pct = explored / len(non_rejected) * 100 if non_rejected else 0

        # Taste stability - improved calculation
        # Compare taste vector to itself over time, not just consecutive vectors
        taste_stability = 0.0
        if len(prefs.taste_vector_history) >= 3:
            # Compare first half average to second half average
            mid = len(prefs.taste_vector_history) // 2
            first_half = np.mean(prefs.taste_vector_history[:mid], axis=0)
            second_half = np.mean(prefs.taste_vector_history[mid:], axis=0)
            first_half = first_half / np.linalg.norm(first_half)
            second_half = second_half / np.linalg.norm(second_half)
            taste_stability = float(np.dot(first_half, second_half))
        elif len(prefs.taste_vector_history) >= 2:
            recent = prefs.taste_vector_history[-2:]
            taste_stability = float(np.dot(recent[0], recent[1]))

        # Taxonomy preferences (archetypes and anchors)
        taxonomy_prefs = self._compute_taxonomy_preferences(prefs)

        # Format archetype insights (sorted by preference)
        archetype_insights = []
        if taxonomy_prefs.get('archetypes'):
            sorted_arch = sorted(
                taxonomy_prefs['archetypes'].items(),
                key=lambda x: x[1]['preference'],
                reverse=True
            )
            for arch, data in sorted_arch:
                archetype_insights.append({
                    'archetype': arch,
                    'preference': data['preference'],
                    'liked_avg': data['liked_avg'],
                    'disliked_avg': data['disliked_avg']
                })

        # Format anchor insights (only significant ones)
        anchor_insights = []
        if taxonomy_prefs.get('anchors'):
            sorted_anchors = sorted(
                taxonomy_prefs['anchors'].items(),
                key=lambda x: abs(x[1]['preference']),
                reverse=True
            )
            for anchor, data in sorted_anchors[:6]:  # Top 6 most significant
                if abs(data['preference']) > 0.01:  # Only if meaningful difference
                    anchor_insights.append({
                        'anchor': anchor,
                        'preference': data['preference'],
                        'direction': 'likes' if data['preference'] > 0 else 'dislikes'
                    })

        return {
            'session_complete': prefs.exploration_complete,
            'total_swipes': prefs.total_swipes,
            'likes': len(prefs.liked_ids),
            'dislikes': len(prefs.disliked_ids),

            'coverage': f"{coverage_pct:.0f}%",
            'clusters_rejected': len(prefs.rejected_regions),
            'taste_stability': round(taste_stability, 3),

            'cluster_health': cluster_health,
            'brand_preferences': top_brands,
            'attribute_preferences': attribute_prefs,

            # NEW: Semantic insights
            'style_profile': {
                'archetypes': archetype_insights,
                'visual_anchors': anchor_insights
            },

            'colors_avoided': list(prefs.colors_to_avoid)
        }


# Quick test
if __name__ == "__main__":
    engine = SwipeEngine()

    user = UserPreferences(
        user_id="test_user",
        colors_to_avoid={"pink", "yellow", "neon"}
    )

    print(f"\n--- Simulating swipes (FEEDBACK-DRIVEN v5 - Semantic) ---")
    print(f"--- Consecutive dislike threshold: {engine.CONSECUTIVE_DISLIKE_THRESHOLD}")
    print(f"--- Cluster rejection at: {engine.CLUSTER_REJECTION_RATIO*100:.0f}% dislike rate ---")
    print(f"--- Learning WITHIN categories using taxonomy signals ---\n")

    swipe_count = 0
    max_swipes = 35

    while swipe_count < max_swipes:
        result = engine.get_next_item(user)
        item_id, session_complete = result

        if session_complete:
            print(f"\n=== SESSION COMPLETE after {swipe_count} swipes ===")
            break

        info = engine.get_item_info(item_id)

        # Simulate: like plain, dislike graphics (strongly)
        is_plain = info.get('category') == 'Plain T-shirts'
        is_small_logo = info.get('category') == 'Small logos'
        is_dark = info['color'] in ['black', 'navy', 'grey', 'white']

        if is_plain and is_dark:
            action = SwipeAction.LIKE
        elif is_small_logo and is_dark:
            action = SwipeAction.LIKE if random.random() > 0.3 else SwipeAction.DISLIKE
        elif is_plain:
            action = SwipeAction.LIKE if random.random() > 0.4 else SwipeAction.DISLIKE
        else:
            # Graphics - mostly dislike
            action = SwipeAction.DISLIKE

        user = engine.record_swipe(user, item_id, action)
        swipe_count += 1

        summary = engine.get_preference_summary(user)
        consec = user.get_consecutive_dislikes()

        print(f"Swipe {swipe_count:2d}: cluster={info['cluster']:2d} | {info.get('category', 'N/A')[:12]:12} | {info['color'][:8]:8} | {'LIKE' if action == SwipeAction.LIKE else 'DISLIKE':7} | clust_rej={summary['clusters_rejected']} consec={consec}")

    print("\n--- LEARNED PREFERENCES ---")
    summary = engine.get_preference_summary(user)

    print(f"\nTotal: {summary['total_swipes']} swipes, {summary['likes']} likes, {summary['dislikes']} dislikes")
    print(f"Coverage: {summary['coverage']}, Rejected clusters: {summary['clusters_rejected']}")
    print(f"Taste stability: {summary['taste_stability']}")

    print("\nCLUSTER HEALTH (with semantic profiles):")
    for c, data in sorted(summary['cluster_health'].items(), key=lambda x: x[1]['health'], reverse=True):
        status = "REJECTED" if data['rejected'] else ""
        profile = data.get('profile', 'unknown')
        print(f"  Cluster {c}: {data['likes']}/{data['likes']+data['dislikes']} liked, health={data['health']:.2f} [{profile}] {status}")

    print("\nSTYLE PROFILE (semantic insights):")
    style = summary.get('style_profile', {})

    print("  Archetypes (preference = liked_avg - disliked_avg):")
    for arch in style.get('archetypes', []):
        direction = "+" if arch['preference'] > 0 else ""
        print(f"    {arch['archetype']}: {direction}{arch['preference']:.3f}")

    print("  Visual Anchors (what catches your eye):")
    for anchor in style.get('visual_anchors', []):
        print(f"    {anchor['anchor']}: {anchor['direction']} ({anchor['preference']:.3f})")

    print("\nATTRIBUTE PREFERENCES:")
    for attr_type, values in summary['attribute_preferences'].items():
        if values:
            print(f"  {attr_type}:")
            for v in values[:3]:
                print(f"    {v['value']}: {v['score']:.2f} ({v['likes']}/{v['total']})")

    print("\nBRAND PREFERENCES:")
    for brand, data in summary['brand_preferences'].items():
        print(f"  {brand}: {data['score']:.2f} ({data['likes']}/{data['total']})")
