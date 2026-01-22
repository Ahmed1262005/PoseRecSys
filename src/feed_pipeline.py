"""
Two-Stage Personalized Feed Pipeline for Fashion Recommendations.

Implements a YouTube-style recommendation system:
- Stage 1: CandidateGenerator (FashionCLIP + Faiss) - ~5ms
- Stage 2: BERT4RecRanker (Sequential patterns) - ~10ms

Total target latency: <100ms for personalized feed generation.
"""
import os
import sys
import time
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.candidate_generator import CandidateGenerator
from src.ranker import BERT4RecRanker, find_latest_bert4rec_checkpoint


class FeedPipeline:
    """
    Two-stage personalized feed generation pipeline.

    Architecture:
    ```
    User Request → Stage 1: Candidate Generation → Stage 2: Ranking → Feed
                   (FashionCLIP + Faiss ANN)       (BERT4Rec)
                   ~1000 candidates                 Top-30
    ```

    Stage 1 uses visual similarity via FashionCLIP embeddings to find
    items similar to what the user has liked. Stage 2 uses BERT4Rec
    to rank these candidates based on sequential patterns in user behavior.
    """

    def __init__(
        self,
        embeddings_path: str = "data/polyvore_u/polyvore_u_clip_embeddings.npy",
        faiss_index_path: str = "models/polyvore_u_faiss.index",
        bert4rec_checkpoint: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the two-stage pipeline.

        Args:
            embeddings_path: Path to FashionCLIP embeddings
            faiss_index_path: Path to Faiss index
            bert4rec_checkpoint: Path to BERT4Rec checkpoint (auto-detect if None)
            device: Device for BERT4Rec ('cuda', 'cpu', or None for auto)
        """
        print("=" * 60)
        print("Initializing Personalized Feed Pipeline")
        print("=" * 60)

        # Stage 1: Candidate Generator
        print("\n--- Stage 1: Candidate Generator ---")
        self.candidate_gen = CandidateGenerator(
            embeddings_path=embeddings_path,
            faiss_index_path=faiss_index_path
        )

        # Stage 2: Ranker
        print("\n--- Stage 2: BERT4Rec Ranker ---")
        if bert4rec_checkpoint is None:
            bert4rec_checkpoint = find_latest_bert4rec_checkpoint()
            if bert4rec_checkpoint is None:
                raise FileNotFoundError(
                    "No BERT4Rec checkpoint found. "
                    "Please train the model first using: python src/train_models.py BERT4Rec"
                )
        self.ranker = BERT4RecRanker(
            checkpoint_path=bert4rec_checkpoint,
            device=device
        )

        print("\n" + "=" * 60)
        print("Pipeline initialized successfully!")
        print("=" * 60)

    def generate_feed(
        self,
        user_id: str,
        user_history: List[int],
        k: int = 30,
        n_candidates: int = 1000,
        return_timing: bool = False
    ) -> List[Tuple[int, float]]:
        """
        Generate personalized feed for a user.

        Two-stage process:
        1. Candidate Generation: Use FashionCLIP + Faiss to find visually
           similar items based on user's recent interactions
        2. Ranking: Use BERT4Rec to score and rank candidates based on
           sequential patterns in user behavior

        Args:
            user_id: User identifier (e.g., 'user_123')
            user_history: List of item IDs the user has interacted with
            k: Number of items to return in feed
            n_candidates: Number of candidates from Stage 1
            return_timing: If True, return (feed, timing_dict)

        Returns:
            List of (item_id, score) tuples sorted by score descending
            If return_timing=True: (feed, {'stage1': ms, 'stage2': ms, 'total': ms})
        """
        timing = {}

        # Stage 1: Candidate Generation
        start = time.time()
        candidates = self.candidate_gen.generate_candidates(
            user_history=user_history,
            k=n_candidates,
            exclude_seen=True
        )
        timing['stage1'] = (time.time() - start) * 1000

        # Stage 2: Ranking
        start = time.time()
        ranked_feed = self.ranker.rank_candidates(
            user_id=user_id,
            candidates=candidates,
            user_history=user_history,
            k=k
        )
        timing['stage2'] = (time.time() - start) * 1000

        timing['total'] = timing['stage1'] + timing['stage2']

        if return_timing:
            return ranked_feed, timing
        return ranked_feed

    def generate_feed_batch(
        self,
        user_ids: List[str],
        user_histories: List[List[int]],
        k: int = 30,
        n_candidates: int = 1000
    ) -> List[List[Tuple[int, float]]]:
        """
        Generate feeds for multiple users.

        Args:
            user_ids: List of user identifiers
            user_histories: List of user histories (item ID lists)
            k: Number of items per user
            n_candidates: Candidates per user

        Returns:
            List of feed lists, one per user
        """
        # Stage 1: Batch candidate generation
        all_candidates = self.candidate_gen.batch_generate_candidates(
            user_histories=user_histories,
            k=n_candidates,
            exclude_seen=True
        )

        # Stage 2: Batch ranking
        all_feeds = self.ranker.batch_rank_candidates(
            user_ids=user_ids,
            candidates_list=all_candidates,
            user_histories=user_histories,
            k=k
        )

        return all_feeds

    def get_feed_with_metadata(
        self,
        user_id: str,
        user_history: List[int],
        k: int = 30,
        image_paths: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Generate feed with item metadata.

        Args:
            user_id: User identifier
            user_history: User's interaction history
            k: Number of items to return
            image_paths: Optional array mapping item_id -> image path

        Returns:
            List of dicts with 'item_id', 'score', and optionally 'image_path'
        """
        feed = self.generate_feed(user_id, user_history, k)

        results = []
        for item_id, score in feed:
            item_data = {
                'item_id': item_id,
                'score': score
            }
            if image_paths is not None and item_id < len(image_paths):
                item_data['image_path'] = str(image_paths[item_id])
            results.append(item_data)

        return results


def main():
    """Test the feed pipeline end-to-end."""
    print("=" * 60)
    print("Testing Feed Pipeline")
    print("=" * 60)

    # Initialize pipeline
    pipeline = FeedPipeline()

    # Test with sample user
    print("\n--- Test 1: Single user feed ---")
    user_id = "test_user"
    user_history = [100, 500, 1000, 5000, 10000]

    feed, timing = pipeline.generate_feed(
        user_id=user_id,
        user_history=user_history,
        k=30,
        return_timing=True
    )

    print(f"User: {user_id}")
    print(f"History: {user_history}")
    print(f"\nTiming:")
    print(f"  Stage 1 (Candidate Gen): {timing['stage1']:.2f}ms")
    print(f"  Stage 2 (Ranking):       {timing['stage2']:.2f}ms")
    print(f"  Total:                   {timing['total']:.2f}ms")
    print(f"\nTop 10 feed items:")
    for i, (item_id, score) in enumerate(feed[:10]):
        print(f"  {i+1}. Item {item_id}: {score:.4f}")

    # Test cold start user
    print("\n--- Test 2: Cold start user ---")
    cold_feed, cold_timing = pipeline.generate_feed(
        user_id="new_user",
        user_history=[],
        k=10,
        return_timing=True
    )
    print(f"Cold start user feed (10 items):")
    print(f"Timing: {cold_timing['total']:.2f}ms")
    for i, (item_id, score) in enumerate(cold_feed[:5]):
        print(f"  {i+1}. Item {item_id}: {score:.4f}")

    # Performance test
    print("\n--- Test 3: Performance benchmark ---")
    n_queries = 50
    histories = [
        list(np.random.randint(1, 100000, size=10))
        for _ in range(n_queries)
    ]

    times = []
    for i, history in enumerate(histories):
        _, timing = pipeline.generate_feed(
            user_id=f"user_{i}",
            user_history=history,
            k=30,
            return_timing=True
        )
        times.append(timing)

    avg_stage1 = np.mean([t['stage1'] for t in times])
    avg_stage2 = np.mean([t['stage2'] for t in times])
    avg_total = np.mean([t['total'] for t in times])
    p95_total = np.percentile([t['total'] for t in times], 95)

    print(f"Performance over {n_queries} queries:")
    print(f"  Avg Stage 1:  {avg_stage1:.2f}ms")
    print(f"  Avg Stage 2:  {avg_stage2:.2f}ms")
    print(f"  Avg Total:    {avg_total:.2f}ms")
    print(f"  P95 Total:    {p95_total:.2f}ms")

    # Load image paths for metadata test
    print("\n--- Test 4: Feed with metadata ---")
    image_paths_file = "data/polyvore_u/all_item_image_paths.npy"
    if os.path.exists(image_paths_file):
        image_paths = np.load(image_paths_file, allow_pickle=True)
        feed_with_meta = pipeline.get_feed_with_metadata(
            user_id="test_user",
            user_history=user_history,
            k=5,
            image_paths=image_paths
        )
        print("Feed with metadata (first 5 items):")
        for item in feed_with_meta:
            print(f"  Item {item['item_id']}: score={item['score']:.4f}, image={item.get('image_path', 'N/A')}")
    else:
        print(f"Skipping metadata test: {image_paths_file} not found")

    print("\n" + "=" * 60)
    print("Feed Pipeline tests completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
