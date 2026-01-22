"""
Ranking-Based Style Learning Engine

Instead of picking 1 favorite, user ranks multiple items.
Provides much richer preference signal per interaction.

Information theory:
- Swipe: 1 bit per interaction
- Four-choice: ~2 bits per interaction
- Ranking 6 items: log2(6!) ≈ 9.5 bits per interaction

Key advantages:
- 4-5x more information per round than four-choice
- Full preference ordering (A > B > C > D > E > F)
- Converges in ~5-6 rounds
- Better captures subtle preferences
"""

import numpy as np
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
import random

from .swipe_engine import SwipeEngine, UserPreferences, SwipeAction


@dataclass
class RankingPreferences(UserPreferences):
    """Extended preferences for ranking model."""

    # Track rounds and rankings
    rounds_completed: int = 0
    ranking_history: List[Dict] = field(default_factory=list)  # {ranking: [...], round: n}

    # Items shown in current round
    current_items: List[str] = field(default_factory=list)

    # Pairwise preference matrix (for advanced analysis)
    # pairwise_wins[cluster_a][cluster_b] = times a ranked above b
    pairwise_wins: Dict = field(default_factory=dict)


class RankingEngine(SwipeEngine):
    """
    Ranking-based selection engine.

    Shows 6 items, user ranks them best to worst.
    Uses weighted contrastive learning based on rank positions.
    """

    # Number of items to rank per round
    ITEMS_PER_ROUND = 6

    # Convergence settings (faster than four-choice due to richer signal)
    MIN_ROUNDS = 4
    MAX_ROUNDS = 10
    TASTE_STABILITY_THRESHOLD = 0.94  # Higher threshold - more data per round

    # Selection strategy
    EARLY_EXPLORATION_ROUNDS = 2  # Pure exploration

    def get_items_to_rank(
        self,
        prefs: RankingPreferences,
        candidates: List[str] = None
    ) -> List[str]:
        """
        Select items for ranking.

        Strategy:
        - Early rounds: 6 items from 6 different clusters (max exploration)
        - Later rounds: 3 taste-based + 3 exploratory (exploit + explore)
        """
        if candidates is None:
            candidates = self.get_candidates(prefs)

        # Filter out already seen items
        available = [c for c in candidates if c not in prefs.seen_ids]

        if len(available) < self.ITEMS_PER_ROUND:
            return available

        selected = []
        used_clusters = set()

        if prefs.rounds_completed < self.EARLY_EXPLORATION_ROUNDS:
            # EXPLORATION PHASE: Pick from different clusters
            selected = self._select_diverse_exploration(prefs, available, used_clusters)
        else:
            # MIXED PHASE: Half taste-based, half exploratory
            selected = self._select_mixed_strategy(prefs, available, used_clusters)

        # Ensure we have enough items (fallback to random)
        while len(selected) < self.ITEMS_PER_ROUND:
            remaining = [c for c in available if c not in selected]
            if not remaining:
                break
            selected.append(random.choice(remaining))

        # Shuffle to avoid position bias
        random.shuffle(selected)

        return selected[:self.ITEMS_PER_ROUND]

    def _select_diverse_exploration(
        self,
        prefs: RankingPreferences,
        available: List[str],
        used_clusters: Set[int]
    ) -> List[str]:
        """Select items from different clusters for exploration."""
        selected = []

        # Get clusters sorted by least explored
        cluster_counts = {}
        for c in range(self.N_VISUAL_CLUSTERS):
            if c not in prefs.rejected_regions:
                cluster_counts[c] = prefs.swipes_per_cluster.get(c, 0)

        sorted_clusters = sorted(cluster_counts.keys(), key=lambda c: cluster_counts[c])

        for cluster_id in sorted_clusters:
            if len(selected) >= self.ITEMS_PER_ROUND:
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

    def _select_mixed_strategy(
        self,
        prefs: RankingPreferences,
        available: List[str],
        used_clusters: Set[int]
    ) -> List[str]:
        """Select half taste-similar, half exploratory items."""
        selected = []

        if prefs.taste_vector is None:
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

        # Pick 3 taste-based items from different clusters
        for item_id, sim in similarities[:30]:  # Top 30 candidates
            if len(selected) >= 3:
                break

            cluster = self.item_to_cluster.get(item_id)
            if cluster not in used_clusters:
                selected.append(item_id)
                used_clusters.add(cluster)

        # Pick 3 exploratory items from underexplored clusters
        cluster_counts = {}
        for c in range(self.N_VISUAL_CLUSTERS):
            if c not in prefs.rejected_regions and c not in used_clusters:
                cluster_counts[c] = prefs.swipes_per_cluster.get(c, 0)

        sorted_clusters = sorted(cluster_counts.keys(), key=lambda c: cluster_counts[c])

        for cluster_id in sorted_clusters:
            if len(selected) >= self.ITEMS_PER_ROUND:
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

    def record_ranking(
        self,
        prefs: RankingPreferences,
        ranked_ids: List[str]
    ) -> RankingPreferences:
        """
        Record user's ranking and update preferences.

        Uses weighted contrastive learning:
        - For each pair (i, j) where i ranked higher than j:
          - Pull taste toward item[i]
          - Push anti-taste toward item[j]
          - Weight by rank distance: |rank_i - rank_j| / max_distance

        Args:
            prefs: User preferences
            ranked_ids: List of item IDs in order (best first, worst last)
        """
        n = len(ranked_ids)

        # Record in history
        prefs.ranking_history.append({
            'round': prefs.rounds_completed,
            'ranking': ranked_ids.copy()
        })

        # Mark all as seen - top half as "liked", bottom half as "disliked"
        mid = n // 2
        for i, item_id in enumerate(ranked_ids):
            if i < mid:
                prefs.liked_ids.append(item_id)
            else:
                prefs.disliked_ids.append(item_id)

        # Update cluster tracking
        for i, item_id in enumerate(ranked_ids):
            cluster = self.item_to_cluster.get(item_id)
            if cluster is not None:
                prefs.swipes_per_cluster[cluster] = prefs.swipes_per_cluster.get(cluster, 0) + 1
                if i < mid:
                    prefs.likes_per_cluster[cluster] = prefs.likes_per_cluster.get(cluster, 0) + 1
                else:
                    prefs.dislikes_per_cluster[cluster] = prefs.dislikes_per_cluster.get(cluster, 0) + 1

        # Update taste vectors using weighted contrastive learning
        self._update_taste_from_ranking(prefs, ranked_ids)

        # Update attribute tracking (weight by rank position)
        for i, item_id in enumerate(ranked_ids):
            # Weight: top ranked items get full weight, bottom get negative
            # Normalize to [-1, 1] range
            position_weight = 1 - (2 * i / (n - 1))  # +1 for first, -1 for last

            item_meta = self.metadata.get(item_id, {})

            if position_weight > 0:  # Top half - record as likes
                for attr_type in ['color', 'fit', 'fabric', 'category']:
                    attr_value = item_meta.get(attr_type, '')
                    if attr_value:
                        if attr_type not in prefs.attribute_likes:
                            prefs.attribute_likes[attr_type] = {}
                        prefs.attribute_likes[attr_type][attr_value] = \
                            prefs.attribute_likes[attr_type].get(attr_value, 0) + position_weight

                # Track brand
                brand = item_meta.get('brand', '')
                if brand:
                    prefs.brand_likes[brand] = prefs.brand_likes.get(brand, 0) + position_weight
            else:  # Bottom half - record as dislikes
                for attr_type in ['color', 'fit', 'fabric', 'category']:
                    attr_value = item_meta.get(attr_type, '')
                    if attr_value:
                        if attr_type not in prefs.attribute_dislikes:
                            prefs.attribute_dislikes[attr_type] = {}
                        prefs.attribute_dislikes[attr_type][attr_value] = \
                            prefs.attribute_dislikes[attr_type].get(attr_value, 0) + abs(position_weight)

        # Track pairwise wins
        for i in range(n):
            for j in range(i + 1, n):
                cluster_i = self.item_to_cluster.get(ranked_ids[i])
                cluster_j = self.item_to_cluster.get(ranked_ids[j])
                if cluster_i is not None and cluster_j is not None:
                    if cluster_i not in prefs.pairwise_wins:
                        prefs.pairwise_wins[cluster_i] = {}
                    prefs.pairwise_wins[cluster_i][cluster_j] = \
                        prefs.pairwise_wins[cluster_i].get(cluster_j, 0) + 1

        # Increment round counter
        prefs.rounds_completed += 1

        # Check for cluster rejection
        for item_id in ranked_ids:
            cluster = self.item_to_cluster.get(item_id)
            if cluster is not None and self._should_reject_cluster(prefs, cluster):
                prefs.rejected_regions.add(cluster)

        return prefs

    def _update_taste_from_ranking(
        self,
        prefs: RankingPreferences,
        ranked_ids: List[str]
    ):
        """
        Update taste vector using weighted pairwise comparisons.

        For each pair (i, j) where i < j (i ranked higher):
        - Move taste toward item[i]
        - Move anti-taste toward item[j]
        - Weight by rank distance
        """
        n = len(ranked_ids)
        max_distance = n - 1

        # Get embeddings
        embeddings = []
        for item_id in ranked_ids:
            emb = self.embeddings_data[item_id]['embedding']
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)

        # Compute weighted positive direction (toward top-ranked items)
        positive_direction = np.zeros_like(embeddings[0])
        negative_direction = np.zeros_like(embeddings[0])
        total_positive_weight = 0
        total_negative_weight = 0

        for i in range(n):
            for j in range(i + 1, n):
                # i is ranked higher than j
                distance = j - i
                weight = distance / max_distance  # Normalize to [0, 1]

                # Add to positive direction (toward higher-ranked item)
                positive_direction += weight * embeddings[i]
                total_positive_weight += weight

                # Add to negative direction (toward lower-ranked item)
                negative_direction += weight * embeddings[j]
                total_negative_weight += weight

        # Normalize directions
        if total_positive_weight > 0:
            positive_direction = positive_direction / total_positive_weight
            positive_direction = positive_direction / np.linalg.norm(positive_direction)

        if total_negative_weight > 0:
            negative_direction = negative_direction / total_negative_weight
            negative_direction = negative_direction / np.linalg.norm(negative_direction)

        # Update taste vector
        # Ranking provides stronger signal, so use higher learning rate
        if prefs.taste_vector is None:
            prefs.taste_vector = positive_direction.copy()
        else:
            # Blend: 60% existing + 40% new (higher than four-choice due to richer signal)
            prefs.taste_vector = 0.6 * prefs.taste_vector + 0.4 * positive_direction
            prefs.taste_vector = prefs.taste_vector / np.linalg.norm(prefs.taste_vector)

        # Update anti-taste vector
        if prefs.anti_taste_vector is None:
            prefs.anti_taste_vector = negative_direction.copy()
        else:
            prefs.anti_taste_vector = 0.7 * prefs.anti_taste_vector + 0.3 * negative_direction
            prefs.anti_taste_vector = prefs.anti_taste_vector / np.linalg.norm(prefs.anti_taste_vector)

        # Track taste history for stability
        prefs.taste_vector_history.append(prefs.taste_vector.copy())
        if len(prefs.taste_vector_history) > 5:
            prefs.taste_vector_history = prefs.taste_vector_history[-5:]

    def record_skip_all(
        self,
        prefs: RankingPreferences,
        shown_ids: List[str]
    ) -> RankingPreferences:
        """Record when user skips all items (none appealing)."""

        # Mark all as mild negative
        for item_id in shown_ids:
            prefs.disliked_ids.append(item_id)

            cluster = self.item_to_cluster.get(item_id)
            if cluster is not None:
                prefs.swipes_per_cluster[cluster] = prefs.swipes_per_cluster.get(cluster, 0) + 1
                prefs.dislikes_per_cluster[cluster] = prefs.dislikes_per_cluster.get(cluster, 0) + 1

        # Update anti-taste with all shown
        all_embs = []
        for item_id in shown_ids:
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

    def is_session_complete(self, prefs: RankingPreferences) -> bool:
        """Check if session should end."""

        if prefs.rounds_completed < self.MIN_ROUNDS:
            return False

        if prefs.rounds_completed >= self.MAX_ROUNDS:
            return True

        # Check taste stability (higher threshold for ranking)
        if len(prefs.taste_vector_history) >= 3:
            recent = prefs.taste_vector_history[-3:]
            similarities = [np.dot(recent[i], recent[i+1]) for i in range(len(recent)-1)]
            if np.mean(similarities) >= self.TASTE_STABILITY_THRESHOLD:
                return True

        # Check coverage
        non_rejected = [c for c in range(self.N_VISUAL_CLUSTERS) if c not in prefs.rejected_regions]
        explored = sum(1 for c in non_rejected if prefs.swipes_per_cluster.get(c, 0) >= 2)

        if explored >= len(non_rejected) * 0.7:
            if len(prefs.taste_vector_history) >= 2:
                return np.dot(prefs.taste_vector_history[-1], prefs.taste_vector_history[-2]) >= 0.92

        return False

    def get_session_summary(self, prefs: RankingPreferences) -> Dict:
        """Get summary of ranking session."""

        base_summary = self.get_preference_summary(prefs)

        # Add ranking-specific stats
        base_summary['rounds_completed'] = prefs.rounds_completed
        base_summary['items_seen'] = len(prefs.liked_ids) + len(prefs.disliked_ids)
        base_summary['rankings_made'] = len(prefs.ranking_history)
        base_summary['items_per_round'] = self.ITEMS_PER_ROUND

        # Information gained estimate
        import math
        items_ranked = prefs.rounds_completed * self.ITEMS_PER_ROUND
        bits_per_ranking = math.log2(math.factorial(self.ITEMS_PER_ROUND))
        base_summary['info_bits_gained'] = prefs.rounds_completed * bits_per_ranking
        base_summary['information_bits'] = base_summary['info_bits_gained']  # Alias for frontend
        base_summary['equivalent_swipes'] = int(base_summary['info_bits_gained'])  # 1 bit per swipe

        # Count pairwise comparisons (n items ranked = n*(n-1)/2 comparisons per round)
        comparisons_per_round = self.ITEMS_PER_ROUND * (self.ITEMS_PER_ROUND - 1) // 2  # 6 items = 15 pairs
        base_summary['pairwise_comparisons'] = prefs.rounds_completed * comparisons_per_round

        # Cluster dominance from pairwise wins
        cluster_scores = {}
        for cluster_i, wins in prefs.pairwise_wins.items():
            total_wins = sum(wins.values())
            cluster_scores[cluster_i] = total_wins

        if cluster_scores:
            sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)
            base_summary['top_clusters'] = sorted_clusters[:3]

        return base_summary


# Quick test
if __name__ == "__main__":
    engine = RankingEngine()

    prefs = RankingPreferences(
        user_id="test_user",
        colors_to_avoid={"pink", "yellow"}
    )

    print(f"=== Ranking Engine Test ===")
    print(f"Total items: {len(engine.item_ids)}")
    print(f"Items per round: {engine.ITEMS_PER_ROUND}")

    import math
    bits_per_round = math.log2(math.factorial(engine.ITEMS_PER_ROUND))
    print(f"Information per round: {bits_per_round:.2f} bits")

    # Simulate a session
    for round_num in range(10):
        items = engine.get_items_to_rank(prefs)

        if len(items) < engine.ITEMS_PER_ROUND:
            print(f"Round {round_num + 1}: Not enough items")
            break

        # Simulate ranking (prefer plain/dark items)
        scores = []
        for item_id in items:
            meta = engine.metadata.get(item_id, {})
            score = random.random() * 0.1  # Base randomness
            if meta.get('category') == 'Plain T-shirts':
                score += 2
            if meta.get('color') in ['black', 'navy', 'grey', 'white']:
                score += 1
            scores.append((item_id, score))

        # Sort by score to get ranking
        scores.sort(key=lambda x: x[1], reverse=True)
        ranked_ids = [item_id for item_id, _ in scores]

        prefs = engine.record_ranking(prefs, ranked_ids)

        top_meta = engine.metadata.get(ranked_ids[0], {})
        print(f"Round {round_num + 1}: Top pick: {top_meta.get('color', 'N/A')} {top_meta.get('category', 'N/A')[:15]}")

        if engine.is_session_complete(prefs):
            print(f"\n=== Session complete after {round_num + 1} rounds ===")
            break

    summary = engine.get_session_summary(prefs)
    print(f"\nTotal rounds: {summary['rounds_completed']}")
    print(f"Items seen: {summary['items_seen']}")
    print(f"Info bits gained: {summary['info_bits_gained']:.1f}")
    print(f"Equivalent swipes: {summary['equivalent_swipes']}")
    print(f"Taste stability: {summary['taste_stability']:.3f}")
