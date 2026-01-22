"""
Candidate Generation using FashionCLIP embeddings and Faiss ANN search.

Stage 1 of the two-stage recommendation system:
- Uses pre-computed FashionCLIP embeddings for visual similarity
- Faiss IndexFlatIP for fast approximate nearest neighbor search
- Generates ~1000 candidates for Stage 2 (BERT4Rec ranking)
"""
import os
import sys
import numpy as np
from typing import List, Optional, Set
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CandidateGenerator:
    """
    Fast candidate retrieval using FashionCLIP + Faiss ANN.

    This is Stage 1 of the YouTube-style two-stage architecture:
    - Retrieves visually similar items based on user's recent interactions
    - Uses mean pooling of item embeddings to build user taste vector
    - Returns ~1000 candidates for ranking by BERT4Rec in Stage 2

    Performance target: ~5ms per query for 200K items
    """

    def __init__(
        self,
        embeddings_path: str = "data/polyvore_u/polyvore_u_clip_embeddings.npy",
        faiss_index_path: str = "models/polyvore_u_faiss.index",
        popular_items_path: Optional[str] = None
    ):
        """
        Initialize candidate generator.

        Args:
            embeddings_path: Path to FashionCLIP embeddings (N, 512)
            faiss_index_path: Path to Faiss index file
            popular_items_path: Optional path to pre-computed popular items for cold start
        """
        try:
            import faiss
        except ImportError:
            print("Installing faiss-cpu...")
            os.system("pip install faiss-cpu")
            import faiss

        self.faiss = faiss

        # Load embeddings
        print(f"Loading embeddings from: {embeddings_path}")
        self.embeddings = np.load(embeddings_path).astype('float32')
        print(f"  Shape: {self.embeddings.shape}")

        # Normalize embeddings (should already be normalized, but ensure)
        faiss.normalize_L2(self.embeddings)

        # Load Faiss index
        print(f"Loading Faiss index from: {faiss_index_path}")
        self.index = faiss.read_index(faiss_index_path)
        print(f"  Index contains {self.index.ntotal} vectors")

        # Validate dimensions match
        assert self.embeddings.shape[0] == self.index.ntotal, \
            f"Mismatch: embeddings has {self.embeddings.shape[0]} items, index has {self.index.ntotal}"

        # Pre-compute popular items for cold start
        self.popular_items = self._load_or_compute_popular_items(popular_items_path)
        print(f"  Popular items for cold start: {len(self.popular_items)} items")

    def _load_or_compute_popular_items(
        self,
        path: Optional[str] = None,
        k: int = 1000
    ) -> List[int]:
        """
        Load or compute popular items for cold start users.

        For now, uses simple heuristic: items with highest average similarity
        to all other items (i.e., most "central" items in embedding space).
        """
        if path and os.path.exists(path):
            return list(np.load(path))

        # Simple heuristic: sample random items for cold start
        # In production, this would be based on interaction frequency
        n_items = self.embeddings.shape[0]

        # Exclude item 0 (padding/empty)
        item_ids = list(range(1, min(n_items, k + 1)))

        return item_ids

    def get_user_vec(self, recent_item_ids: List[int]) -> Optional[np.ndarray]:
        """
        Build user taste vector from recent interactions.

        Uses mean pooling of item embeddings to create a single
        user representation vector.

        Args:
            recent_item_ids: List of item IDs user recently interacted with

        Returns:
            User vector (1, 512) normalized for cosine similarity, or None if empty
        """
        if not recent_item_ids:
            return None

        # Filter valid item IDs
        valid_ids = [i for i in recent_item_ids if 0 <= i < len(self.embeddings)]

        if not valid_ids:
            return None

        # Get embeddings for recent items
        item_embs = self.embeddings[valid_ids]

        # Mean pooling
        user_vec = np.mean(item_embs, axis=0, keepdims=True).astype('float32')

        # Normalize for cosine similarity (Inner Product after normalization)
        self.faiss.normalize_L2(user_vec)

        return user_vec

    def generate_candidates(
        self,
        user_history: List[int],
        k: int = 1000,
        exclude_seen: bool = True
    ) -> List[int]:
        """
        Stage 1: Fast candidate retrieval using Faiss ANN.

        Args:
            user_history: List of item_ids user recently interacted with
            k: Number of candidates to return
            exclude_seen: Whether to exclude already seen items

        Returns:
            List of candidate item_ids (length <= k)
        """
        user_vec = self.get_user_vec(user_history)

        # Cold start: return popular items
        if user_vec is None:
            return self.popular_items[:k]

        # Determine how many to retrieve (extra for filtering)
        n_retrieve = k + len(user_history) if exclude_seen else k
        n_retrieve = min(n_retrieve, self.index.ntotal)

        # Faiss ANN search
        distances, indices = self.index.search(user_vec, n_retrieve)

        # Process results
        candidates = []
        seen = set(user_history) if exclude_seen else set()

        for idx in indices[0]:
            if idx == 0:  # Skip padding item
                continue
            if exclude_seen and idx in seen:
                continue
            candidates.append(int(idx))
            if len(candidates) >= k:
                break

        return candidates

    def generate_candidates_with_scores(
        self,
        user_history: List[int],
        k: int = 1000,
        exclude_seen: bool = True
    ) -> List[tuple]:
        """
        Generate candidates with similarity scores.

        Args:
            user_history: List of item_ids user recently interacted with
            k: Number of candidates to return
            exclude_seen: Whether to exclude already seen items

        Returns:
            List of (item_id, similarity_score) tuples
        """
        user_vec = self.get_user_vec(user_history)

        # Cold start: return popular items with score 0
        if user_vec is None:
            return [(item_id, 0.0) for item_id in self.popular_items[:k]]

        # Determine how many to retrieve
        n_retrieve = k + len(user_history) if exclude_seen else k
        n_retrieve = min(n_retrieve, self.index.ntotal)

        # Faiss ANN search
        distances, indices = self.index.search(user_vec, n_retrieve)

        # Process results
        candidates = []
        seen = set(user_history) if exclude_seen else set()

        for idx, score in zip(indices[0], distances[0]):
            if idx == 0:  # Skip padding item
                continue
            if exclude_seen and idx in seen:
                continue
            candidates.append((int(idx), float(score)))
            if len(candidates) >= k:
                break

        return candidates

    def batch_generate_candidates(
        self,
        user_histories: List[List[int]],
        k: int = 1000,
        exclude_seen: bool = True
    ) -> List[List[int]]:
        """
        Batch candidate generation for multiple users.

        Args:
            user_histories: List of user histories (each is a list of item_ids)
            k: Number of candidates per user
            exclude_seen: Whether to exclude seen items

        Returns:
            List of candidate lists, one per user
        """
        results = []

        # Build user vectors for non-cold-start users
        user_vecs = []
        user_indices = []  # Track which users have valid vectors

        for i, history in enumerate(user_histories):
            user_vec = self.get_user_vec(history)
            if user_vec is not None:
                user_vecs.append(user_vec)
                user_indices.append(i)

        # Batch search for users with valid vectors
        if user_vecs:
            batch_vecs = np.vstack(user_vecs)
            n_retrieve = k + 100  # Extra buffer for filtering
            n_retrieve = min(n_retrieve, self.index.ntotal)

            distances, indices = self.index.search(batch_vecs, n_retrieve)

            # Build index mapping for batch results
            batch_results = {}
            for batch_idx, user_idx in enumerate(user_indices):
                history = user_histories[user_idx]
                seen = set(history) if exclude_seen else set()

                candidates = []
                for idx in indices[batch_idx]:
                    if idx == 0:
                        continue
                    if exclude_seen and idx in seen:
                        continue
                    candidates.append(int(idx))
                    if len(candidates) >= k:
                        break

                batch_results[user_idx] = candidates
        else:
            batch_results = {}

        # Assemble final results
        for i, history in enumerate(user_histories):
            if i in batch_results:
                results.append(batch_results[i])
            else:
                # Cold start
                results.append(self.popular_items[:k])

        return results


def main():
    """Test candidate generation."""
    print("=" * 60)
    print("Testing Candidate Generator")
    print("=" * 60)

    # Initialize
    generator = CandidateGenerator(
        embeddings_path='data/polyvore_u/polyvore_u_clip_embeddings.npy',
        faiss_index_path='models/polyvore_u_faiss.index'
    )

    # Test with sample user history
    print("\n--- Test 1: Single user with history ---")
    user_history = [100, 500, 1000, 5000, 10000]

    import time
    start = time.time()
    candidates = generator.generate_candidates(user_history, k=100)
    elapsed = (time.time() - start) * 1000

    print(f"User history: {user_history}")
    print(f"Generated {len(candidates)} candidates in {elapsed:.2f}ms")
    print(f"First 10 candidates: {candidates[:10]}")

    # Test with scores
    print("\n--- Test 2: Candidates with scores ---")
    candidates_with_scores = generator.generate_candidates_with_scores(user_history, k=10)
    print("Top 10 candidates with similarity scores:")
    for item_id, score in candidates_with_scores:
        print(f"  Item {item_id}: {score:.4f}")

    # Test cold start
    print("\n--- Test 3: Cold start user ---")
    cold_start_candidates = generator.generate_candidates([], k=10)
    print(f"Cold start candidates: {cold_start_candidates}")

    # Test batch generation
    print("\n--- Test 4: Batch generation ---")
    user_histories = [
        [100, 200, 300],
        [1000, 2000, 3000],
        [],  # Cold start user
        [50000, 60000, 70000]
    ]

    start = time.time()
    batch_candidates = generator.batch_generate_candidates(user_histories, k=50)
    elapsed = (time.time() - start) * 1000

    print(f"Batch generated candidates for {len(user_histories)} users in {elapsed:.2f}ms")
    for i, candidates in enumerate(batch_candidates):
        print(f"  User {i}: {len(candidates)} candidates, first 5: {candidates[:5]}")

    # Performance test
    print("\n--- Test 5: Performance benchmark ---")
    n_queries = 1000
    random_histories = [
        list(np.random.randint(1, 100000, size=10))
        for _ in range(n_queries)
    ]

    start = time.time()
    for history in random_histories:
        generator.generate_candidates(history, k=1000)
    elapsed = (time.time() - start) * 1000

    print(f"{n_queries} queries completed in {elapsed:.2f}ms")
    print(f"Average latency: {elapsed/n_queries:.3f}ms per query")

    print("\n" + "=" * 60)
    print("Candidate Generator tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
