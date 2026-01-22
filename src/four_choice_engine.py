"""
Four-Choice Style Learning Engine

Instead of Tinder-style swipes, show 4 items at once.
User picks their favorite - provides richer preference signal.

Key advantages:
- 2+ bits per interaction (vs 1 bit for swipe)
- Relative preference (A > B, C, D)
- Faster convergence (~10 rounds vs 40 swipes)
- Less decision fatigue
"""

import numpy as np
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
import random

from swipe_engine import SwipeEngine, UserPreferences, SwipeAction


@dataclass
class FourChoicePreferences(UserPreferences):
    """Extended preferences for 4-choice model."""

    # Track rounds and choices
    rounds_completed: int = 0
    choice_history: List[Dict] = field(default_factory=list)  # {winner, losers, round}

    # Items shown in current round (for recording)
    current_four: List[str] = field(default_factory=list)


class FourChoiceEngine(SwipeEngine):
    """
    Four-choice selection engine.

    Shows 4 items, user picks favorite.
    Uses contrastive learning to update taste vector.
    """

    # Convergence settings (faster than swipe)
    MIN_ROUNDS = 8
    MAX_ROUNDS = 15
    TASTE_STABILITY_THRESHOLD = 0.92

    # Selection strategy
    EARLY_EXPLORATION_ROUNDS = 3  # Pure cluster exploration

    # Contrast dimensions for strategic selection
    CONTRAST_DIMENSIONS = ['archetype', 'category', 'color_family', 'fit']

    # Color families for contrast
    COLOR_FAMILIES = {
        'dark': {'black', 'navy', 'dark grey', 'dark navy', 'charcoal'},
        'light': {'white', 'cream', 'beige', 'off-white', 'light grey', 'sand', 'oat milk'},
        'cool': {'blue', 'light blue', 'navy', 'green', 'teal', 'purple'},
        'warm': {'red', 'orange', 'yellow', 'pink', 'coral', 'burnt orange'},
        'earth': {'brown', 'olive', 'tan', 'khaki', 'olive green', 'forest green'},
    }

    def _get_color_family(self, color: str) -> str:
        """Map a color to its family for contrast purposes."""
        color_lower = color.lower() if color else ''
        for family, colors in self.COLOR_FAMILIES.items():
            if any(c in color_lower for c in colors):
                return family
        return 'neutral'

    def _get_item_contrast_profile(self, item_id: str) -> Dict:
        """Get contrast-relevant attributes for an item."""
        meta = self.metadata.get(item_id, {})
        return {
            'archetype': meta.get('archetype', 'unknown'),
            'category': meta.get('category', ''),
            'color_family': self._get_color_family(meta.get('color', '')),
            'fit': meta.get('fit', 'Regular'),
        }

    def _calculate_contrast_score(self, selected: List[str], candidate: str) -> float:
        """
        Calculate how much contrast a candidate adds to the selection.
        Higher score = more contrast = better.
        """
        if not selected:
            return 1.0

        candidate_profile = self._get_item_contrast_profile(candidate)
        contrast_score = 0.0

        for existing_id in selected:
            existing_profile = self._get_item_contrast_profile(existing_id)

            # Score each dimension
            for dim in self.CONTRAST_DIMENSIONS:
                if candidate_profile[dim] != existing_profile[dim]:
                    # Different value = contrast
                    contrast_score += 1.0

        # Normalize by number of comparisons
        return contrast_score / (len(selected) * len(self.CONTRAST_DIMENSIONS))

    def get_four_items(
        self,
        prefs: FourChoicePreferences,
        candidates: List[str] = None
    ) -> List[str]:
        """
        Select 4 contrasting items for comparison.

        Strategy: Pick items that CONTRAST on key dimensions:
        - Archetype (classic vs dramatic vs creative vs natural)
        - Category (Plain vs Graphics vs Small logos vs Athletic)
        - Color family (dark vs light vs cool vs warm)
        - Fit (slim vs regular vs relaxed)

        This maximizes information gain from user's choice.
        """
        if candidates is None:
            candidates = self.get_candidates(prefs)

        # Filter out already shown items
        available = [c for c in candidates if c not in prefs.seen_ids]

        if len(available) < 4:
            return available

        selected = []

        if prefs.rounds_completed < self.EARLY_EXPLORATION_ROUNDS:
            # EXPLORATION: Maximize contrast on archetype + category
            selected = self._select_max_contrast(prefs, available, focus_dims=['archetype', 'category'])
        else:
            # EXPLOITATION + EXPLORATION: Mix taste-aligned with contrasting
            selected = self._select_contrast_with_taste(prefs, available)

        # Ensure we have 4 items
        while len(selected) < 4:
            remaining = [c for c in available if c not in selected]
            if not remaining:
                break
            # Pick item with max contrast to current selection
            best_candidate = max(remaining, key=lambda c: self._calculate_contrast_score(selected, c))
            selected.append(best_candidate)

        # Shuffle to avoid position bias
        random.shuffle(selected)
        return selected

    def _select_max_contrast(
        self,
        prefs: FourChoicePreferences,
        available: List[str],
        focus_dims: List[str] = None
    ) -> List[str]:
        """Select 4 items maximizing contrast on specified dimensions."""
        if focus_dims is None:
            focus_dims = self.CONTRAST_DIMENSIONS

        selected = []
        used_values = {dim: set() for dim in focus_dims}

        # Group items by their contrast profile
        for item_id in available:
            if len(selected) >= 4:
                break

            profile = self._get_item_contrast_profile(item_id)

            # Check if this item adds new values on focus dimensions
            adds_contrast = False
            for dim in focus_dims:
                if profile[dim] not in used_values[dim]:
                    adds_contrast = True
                    break

            if adds_contrast or len(selected) == 0:
                selected.append(item_id)
                for dim in focus_dims:
                    used_values[dim].add(profile[dim])

        return selected

    def _select_contrast_with_taste(
        self,
        prefs: FourChoicePreferences,
        available: List[str]
    ) -> List[str]:
        """Select items: 1 taste-aligned + 3 contrasting alternatives."""
        selected = []

        if prefs.taste_vector is None:
            return self._select_max_contrast(prefs, available)

        # Find taste-aligned item
        similarities = []
        for item_id in available:
            emb = self.embeddings_data[item_id]['embedding']
            emb_norm = emb / np.linalg.norm(emb)
            sim = np.dot(emb_norm, prefs.taste_vector)
            if prefs.anti_taste_vector is not None:
                anti_sim = np.dot(emb_norm, prefs.anti_taste_vector)
                sim = sim - 0.3 * anti_sim
            similarities.append((item_id, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)

        # Pick 1 taste-aligned item
        if similarities:
            selected.append(similarities[0][0])

        # Pick 3 items that contrast with the taste-aligned item
        remaining = [item_id for item_id, _ in similarities[1:] if item_id not in selected]

        # Use contrast scoring to pick diverse alternatives
        for _ in range(3):
            if not remaining:
                break
            # Find item with max contrast to current selection
            best_idx = 0
            best_score = -1
            for i, item_id in enumerate(remaining):
                score = self._calculate_contrast_score(selected, item_id)
                if score > best_score:
                    best_score = score
                    best_idx = i

            selected.append(remaining[best_idx])
            remaining.pop(best_idx)

        return selected

    def _select_diverse_exploration(
        self,
        prefs: FourChoicePreferences,
        available: List[str],
        used_clusters: Set[int]
    ) -> List[str]:
        """Select 4 items from 4 different clusters for exploration."""
        selected = []

        # Get clusters sorted by least explored
        cluster_counts = {}
        for c in range(self.N_VISUAL_CLUSTERS):
            if c not in prefs.rejected_regions:
                cluster_counts[c] = prefs.swipes_per_cluster.get(c, 0)

        sorted_clusters = sorted(cluster_counts.keys(), key=lambda c: cluster_counts[c])

        for cluster_id in sorted_clusters:
            if len(selected) >= 4:
                break

            cluster_candidates = [
                c for c in available
                if self.item_to_cluster.get(c) == cluster_id
                and c not in selected
            ]

            if cluster_candidates:
                # Pick randomly from cluster
                selected.append(random.choice(cluster_candidates))
                used_clusters.add(cluster_id)

        return selected

    def _select_mixed_strategy(
        self,
        prefs: FourChoicePreferences,
        available: List[str],
        used_clusters: Set[int]
    ) -> List[str]:
        """Select 2 taste-similar + 2 exploratory items."""
        selected = []

        if prefs.taste_vector is None:
            # No taste yet, fall back to exploration
            return self._select_diverse_exploration(prefs, available, used_clusters)

        # Compute similarities to taste vector
        similarities = []
        for item_id in available:
            emb = self.embeddings_data[item_id]['embedding']
            emb_norm = emb / np.linalg.norm(emb)
            sim = np.dot(emb_norm, prefs.taste_vector)

            # Penalty for similarity to anti-taste
            if prefs.anti_taste_vector is not None:
                anti_sim = np.dot(emb_norm, prefs.anti_taste_vector)
                sim = sim - 0.3 * anti_sim

            similarities.append((item_id, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Pick 2 taste-based items from different clusters
        for item_id, sim in similarities[:20]:  # Top 20 candidates
            if len(selected) >= 2:
                break

            cluster = self.item_to_cluster.get(item_id)
            if cluster not in used_clusters:
                selected.append(item_id)
                used_clusters.add(cluster)

        # Pick 2 exploratory items from underexplored clusters
        cluster_counts = {}
        for c in range(self.N_VISUAL_CLUSTERS):
            if c not in prefs.rejected_regions and c not in used_clusters:
                cluster_counts[c] = prefs.swipes_per_cluster.get(c, 0)

        sorted_clusters = sorted(cluster_counts.keys(), key=lambda c: cluster_counts[c])

        for cluster_id in sorted_clusters:
            if len(selected) >= 4:
                break

            cluster_candidates = [
                c for c in available
                if self.item_to_cluster.get(c) == cluster_id
                and c not in selected
            ]

            if cluster_candidates:
                selected.append(random.choice(cluster_candidates))
                used_clusters.add(cluster_id)

        return selected

    def record_choice(
        self,
        prefs: FourChoicePreferences,
        winner_id: str,
        all_shown: List[str]
    ) -> FourChoicePreferences:
        """
        Record user's choice and update preferences.

        Uses contrastive learning:
        - Winner pulls taste vector toward it
        - Losers push anti-taste vector toward them
        """
        losers = [item_id for item_id in all_shown if item_id != winner_id]

        # Record in history
        prefs.choice_history.append({
            'round': prefs.rounds_completed,
            'winner': winner_id,
            'losers': losers
        })

        # Mark all as seen
        prefs.liked_ids.append(winner_id)
        for loser in losers:
            # Losers go to disliked (they were explicitly not chosen)
            prefs.disliked_ids.append(loser)

        # Update cluster tracking
        for item_id in all_shown:
            cluster = self.item_to_cluster.get(item_id)
            if cluster is not None:
                prefs.swipes_per_cluster[cluster] = prefs.swipes_per_cluster.get(cluster, 0) + 1
                if item_id == winner_id:
                    prefs.likes_per_cluster[cluster] = prefs.likes_per_cluster.get(cluster, 0) + 1
                else:
                    prefs.dislikes_per_cluster[cluster] = prefs.dislikes_per_cluster.get(cluster, 0) + 1

        # Update taste vectors using contrastive learning
        self._update_taste_contrastive(prefs, winner_id, losers)

        # Update attribute tracking
        winner_meta = self.metadata.get(winner_id, {})
        for attr_type in ['color', 'fit', 'fabric', 'category']:
            attr_value = winner_meta.get(attr_type, '')
            if attr_value:
                if attr_type not in prefs.attribute_likes:
                    prefs.attribute_likes[attr_type] = {}
                prefs.attribute_likes[attr_type][attr_value] = \
                    prefs.attribute_likes[attr_type].get(attr_value, 0) + 1

        # Track brand
        brand = winner_meta.get('brand', '')
        if brand:
            prefs.brand_likes[brand] = prefs.brand_likes.get(brand, 0) + 1

        # Increment round counter
        prefs.rounds_completed += 1

        # Check for cluster rejection
        for cluster in set(self.item_to_cluster.get(item_id) for item_id in all_shown):
            if cluster is not None and self._should_reject_cluster(prefs, cluster):
                prefs.rejected_regions.add(cluster)

        return prefs

    def _update_taste_contrastive(
        self,
        prefs: FourChoicePreferences,
        winner_id: str,
        loser_ids: List[str]
    ):
        """Update taste vector using contrastive learning."""

        # Get winner embedding
        winner_emb = self.embeddings_data[winner_id]['embedding']
        winner_emb = winner_emb / np.linalg.norm(winner_emb)

        # Get loser embeddings
        loser_embs = []
        for loser_id in loser_ids:
            emb = self.embeddings_data[loser_id]['embedding']
            loser_embs.append(emb / np.linalg.norm(emb))
        loser_mean = np.mean(loser_embs, axis=0)
        loser_mean = loser_mean / np.linalg.norm(loser_mean)

        # Update taste vector (move toward winner)
        if prefs.taste_vector is None:
            prefs.taste_vector = winner_emb.copy()
        else:
            # Weighted average: 70% existing + 30% new winner
            prefs.taste_vector = 0.7 * prefs.taste_vector + 0.3 * winner_emb
            prefs.taste_vector = prefs.taste_vector / np.linalg.norm(prefs.taste_vector)

        # Update anti-taste vector (move toward losers, lighter weight)
        if prefs.anti_taste_vector is None:
            prefs.anti_taste_vector = loser_mean.copy()
        else:
            # Weighted average: 80% existing + 20% new losers
            prefs.anti_taste_vector = 0.8 * prefs.anti_taste_vector + 0.2 * loser_mean
            prefs.anti_taste_vector = prefs.anti_taste_vector / np.linalg.norm(prefs.anti_taste_vector)

        # Track taste history for stability
        prefs.taste_vector_history.append(prefs.taste_vector.copy())
        if len(prefs.taste_vector_history) > 5:
            prefs.taste_vector_history = prefs.taste_vector_history[-5:]

    def record_skip_all(
        self,
        prefs: FourChoicePreferences,
        all_shown: List[str]
    ) -> FourChoicePreferences:
        """Record when user skips all 4 (none appealing)."""

        # Mark all as mild negative
        for item_id in all_shown:
            prefs.disliked_ids.append(item_id)

            cluster = self.item_to_cluster.get(item_id)
            if cluster is not None:
                prefs.swipes_per_cluster[cluster] = prefs.swipes_per_cluster.get(cluster, 0) + 1
                prefs.dislikes_per_cluster[cluster] = prefs.dislikes_per_cluster.get(cluster, 0) + 1

        # Update anti-taste with all shown
        all_embs = []
        for item_id in all_shown:
            emb = self.embeddings_data[item_id]['embedding']
            all_embs.append(emb / np.linalg.norm(emb))

        all_mean = np.mean(all_embs, axis=0)
        all_mean = all_mean / np.linalg.norm(all_mean)

        if prefs.anti_taste_vector is None:
            prefs.anti_taste_vector = all_mean
        else:
            prefs.anti_taste_vector = 0.7 * prefs.anti_taste_vector + 0.3 * all_mean
            prefs.anti_taste_vector = prefs.anti_taste_vector / np.linalg.norm(prefs.anti_taste_vector)

        prefs.rounds_completed += 1

        return prefs

    def is_session_complete(self, prefs: FourChoicePreferences) -> bool:
        """Check if session should end."""

        if prefs.rounds_completed < self.MIN_ROUNDS:
            return False

        if prefs.rounds_completed >= self.MAX_ROUNDS:
            return True

        # Check taste stability
        if len(prefs.taste_vector_history) >= 3:
            recent = prefs.taste_vector_history[-3:]
            similarities = [np.dot(recent[i], recent[i+1]) for i in range(len(recent)-1)]
            if np.mean(similarities) >= self.TASTE_STABILITY_THRESHOLD:
                return True

        # Check coverage of non-rejected clusters
        non_rejected = [c for c in range(self.N_VISUAL_CLUSTERS) if c not in prefs.rejected_regions]
        explored = sum(1 for c in non_rejected if prefs.swipes_per_cluster.get(c, 0) >= 2)

        if explored >= len(non_rejected) * 0.7:
            # Good coverage, check stability
            if len(prefs.taste_vector_history) >= 2:
                return np.dot(prefs.taste_vector_history[-1], prefs.taste_vector_history[-2]) >= 0.90

        return False

    def get_session_summary(self, prefs: FourChoicePreferences) -> Dict:
        """Get summary of 4-choice session."""

        base_summary = self.get_preference_summary(prefs)

        # Add 4-choice specific stats
        base_summary['rounds_completed'] = prefs.rounds_completed
        base_summary['items_seen'] = len(prefs.liked_ids) + len(prefs.disliked_ids)
        base_summary['choices_made'] = len(prefs.choice_history)

        # Win rate per cluster
        cluster_wins = {}
        for choice in prefs.choice_history:
            winner_cluster = self.item_to_cluster.get(choice['winner'])
            if winner_cluster is not None:
                if winner_cluster not in cluster_wins:
                    cluster_wins[winner_cluster] = {'wins': 0, 'shown': 0}
                cluster_wins[winner_cluster]['wins'] += 1

            for loser in choice['losers']:
                loser_cluster = self.item_to_cluster.get(loser)
                if loser_cluster is not None:
                    if loser_cluster not in cluster_wins:
                        cluster_wins[loser_cluster] = {'wins': 0, 'shown': 0}
                    cluster_wins[loser_cluster]['shown'] += 1

        # Calculate win rates
        for cluster, data in cluster_wins.items():
            total = data['wins'] + data['shown']
            data['win_rate'] = data['wins'] / total if total > 0 else 0

        base_summary['cluster_win_rates'] = cluster_wins

        return base_summary


# Quick test
if __name__ == "__main__":
    engine = FourChoiceEngine()

    prefs = FourChoicePreferences(
        user_id="test_user",
        colors_to_avoid={"pink", "yellow"}
    )

    print(f"=== Four-Choice Engine Test ===")
    print(f"Total items: {len(engine.item_ids)}")

    # Simulate a session
    for round_num in range(12):
        four_items = engine.get_four_items(prefs)

        if len(four_items) < 4:
            print(f"Round {round_num + 1}: Not enough items")
            break

        # Simulate choosing (prefer plain/dark items)
        scores = []
        for item_id in four_items:
            meta = engine.metadata.get(item_id, {})
            score = 0
            if meta.get('category') == 'Plain T-shirts':
                score += 2
            if meta.get('color') in ['black', 'navy', 'grey', 'white']:
                score += 1
            scores.append((item_id, score))

        # Pick highest score (with some randomness)
        scores.sort(key=lambda x: x[1] + random.random() * 0.5, reverse=True)
        winner = scores[0][0]

        prefs = engine.record_choice(prefs, winner, four_items)

        winner_meta = engine.metadata.get(winner, {})
        print(f"Round {round_num + 1}: Chose {winner_meta.get('category', 'N/A')[:15]} / {winner_meta.get('color', 'N/A')}")

        if engine.is_session_complete(prefs):
            print(f"\n=== Session complete after {round_num + 1} rounds ===")
            break

    summary = engine.get_session_summary(prefs)
    print(f"\nTotal rounds: {summary['rounds_completed']}")
    print(f"Items seen: {summary['items_seen']}")
    print(f"Likes: {summary['likes']}, Dislikes: {summary['dislikes']}")
    print(f"Taste stability: {summary['taste_stability']:.3f}")
