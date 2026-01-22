"""
Hybrid Fashion Feed Generator
Combines FashionCLIP (style-aware) + DuoRec (behavior-aware) for recommendations.

Architecture:
1. FashionCLIP: Visual similarity for candidate generation
2. DuoRec: Sequential behavior modeling for ranking
"""

import sys
import os
# Use standard RecBole, not DuoRec fork
if '/home/ubuntu/recSys/DuoRec' in sys.path:
    sys.path.remove('/home/ubuntu/recSys/DuoRec')

import torch
import numpy as np
import faiss
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


class AmazonFashionFeed:
    """
    Hybrid recommendation system for Amazon Fashion.

    Flow:
    1. Build user CLIP vector from their recent interactions
    2. Get ~1000 candidates via Faiss nearest neighbors
    3. Rank candidates with DuoRec sequential model
    4. Return top-k items
    """

    def __init__(
        self,
        embeddings_path: str = "data/amazon_fashion/processed/amazon_mens_embeddings.pkl",
        faiss_index_path: str = "models/amazon_mens_faiss.index",
        faiss_ids_path: str = "models/amazon_mens_faiss_ids.npy",
        duorec_checkpoint: str = None,  # Will be set when training completes
        user_interactions_path: str = "data/amazon_fashion/recbole/amazon_mens/amazon_mens.inter",
        use_gpu: bool = True
    ):
        self.use_gpu = use_gpu
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

        print("Loading Amazon Fashion Feed components...", flush=True)

        # Load FashionCLIP embeddings
        print("  Loading FashionCLIP embeddings...", flush=True)
        with open(embeddings_path, 'rb') as f:
            self.embeddings_dict = pickle.load(f)

        # Build embedding matrix and ID mapping
        self.item_ids = list(self.embeddings_dict.keys())
        self.id_to_idx = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        self.embeddings_matrix = np.vstack([
            self.embeddings_dict[item_id] for item_id in self.item_ids
        ]).astype('float32')
        print(f"  Loaded {len(self.item_ids)} embeddings", flush=True)

        # Load Faiss index
        print("  Loading Faiss index...", flush=True)
        self.faiss_index = faiss.read_index(faiss_index_path)
        self.faiss_ids = np.load(faiss_ids_path, allow_pickle=True)

        if use_gpu and faiss.get_num_gpus() > 0:
            gpu_res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(gpu_res, 0, self.faiss_index)
        print("  Faiss index ready", flush=True)

        # Load user interaction history
        print("  Loading user interactions...", flush=True)
        self.user_history = self._load_user_history(user_interactions_path)
        print(f"  Loaded {len(self.user_history)} user histories", flush=True)

        # Sequential model (SASRec)
        self.seq_model = None
        self.seq_dataset = None
        self.seq_config = None

        if duorec_checkpoint and os.path.exists(duorec_checkpoint):
            print("  Loading SASRec model...", flush=True)
            self._load_sasrec(duorec_checkpoint)

        print(f"Feed initialized with {len(self.item_ids)} items, {len(self.user_history)} users", flush=True)

    def _load_user_history(self, interactions_path: str) -> Dict[str, List[str]]:
        """Load user interaction history from RecBole format."""
        user_history = defaultdict(list)

        with open(interactions_path, 'r') as f:
            header = f.readline()  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    user_id, item_id, timestamp = parts[0], parts[1], float(parts[2])
                    user_history[user_id].append((item_id, timestamp))

        # Sort by timestamp and extract just item IDs
        for user_id in user_history:
            user_history[user_id] = [
                item_id for item_id, ts in sorted(user_history[user_id], key=lambda x: x[1])
            ]

        return dict(user_history)

    def _load_sasrec(self, checkpoint_path: str):
        """Load SASRec model and dataset."""
        try:
            from recbole.config import Config
            from recbole.data import create_dataset
            from recbole.model.sequential_recommender import SASRec

            config_dict = {
                'data_path': '/home/ubuntu/recSys/outfitTransformer/data/amazon_fashion/recbole',
                'dataset': 'amazon_mens',
                'USER_ID_FIELD': 'user_id',
                'ITEM_ID_FIELD': 'item_id',
                'TIME_FIELD': 'timestamp',
                'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
                'MAX_ITEM_LIST_LENGTH': 50,
                'train_neg_sample_args': None,  # CE loss doesn't need negative sampling
                'n_layers': 4,
                'n_heads': 8,
                'hidden_size': 256,
                'inner_size': 1024,
                'hidden_dropout_prob': 0.3,
                'attn_dropout_prob': 0.3,
                'hidden_act': 'gelu',
                'layer_norm_eps': 1e-12,
                'loss_type': 'CE',
            }

            config = Config(model='SASRec', dataset='amazon_mens', config_dict=config_dict)
            self.seq_dataset = create_dataset(config)
            self.seq_config = config

            self.seq_model = SASRec(config, self.seq_dataset).to(self.device)
            # Fix for PyTorch 2.6 weights_only issue
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.seq_model.load_state_dict(state['state_dict'])
            self.seq_model.eval()

            print(f"SASRec model loaded from {checkpoint_path}")

        except Exception as e:
            print(f"Warning: Could not load SASRec model: {e}")
            import traceback
            traceback.print_exc()
            self.seq_model = None

    def build_user_clip_vector(self, item_ids: List[str]) -> Optional[np.ndarray]:
        """
        Build user embedding by averaging FashionCLIP vectors of interacted items.

        Args:
            item_ids: List of item IDs the user has interacted with

        Returns:
            Normalized user embedding vector (512,) or None if no valid items
        """
        valid_embeddings = []

        for item_id in item_ids:
            if item_id in self.embeddings_dict:
                valid_embeddings.append(self.embeddings_dict[item_id])

        if not valid_embeddings:
            return None

        # Average and normalize
        user_vec = np.mean(valid_embeddings, axis=0).astype('float32')
        user_vec = user_vec / (np.linalg.norm(user_vec) + 1e-8)

        return user_vec

    def get_clip_candidates(
        self,
        user_vec: np.ndarray,
        k: int = 1000,
        exclude_items: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Get candidate items using FashionCLIP similarity.

        Args:
            user_vec: User embedding vector
            k: Number of candidates to retrieve
            exclude_items: Items to exclude (e.g., already seen)

        Returns:
            List of (item_id, similarity_score) tuples
        """
        # Search more than k to account for exclusions (need larger buffer for big histories)
        exclude_count = len(exclude_items) if exclude_items else 0
        search_k = k + exclude_count + 100 if exclude_items else k

        query = user_vec.reshape(1, -1)
        D, I = self.faiss_index.search(query, search_k)

        exclude_set = set(exclude_items) if exclude_items else set()
        candidates = []

        for dist, idx in zip(D[0], I[0]):
            item_id = self.faiss_ids[idx]
            if item_id not in exclude_set:
                candidates.append((item_id, float(dist)))
                if len(candidates) >= k:
                    break

        return candidates

    def rank_with_sasrec(
        self,
        user_id: str,
        candidate_items: List[str],
        topk: int = 50
    ) -> List[Tuple[str, float]]:
        """
        Rank candidates using SASRec sequential model.

        Args:
            user_id: User identifier
            candidate_items: List of candidate item IDs
            topk: Number of top items to return

        Returns:
            Ranked list of (item_id, score) tuples
        """
        if self.seq_model is None:
            # Fallback: return candidates as-is (CLIP ranking)
            return [(item_id, 0.0) for item_id in candidate_items[:topk]]

        try:
            # Get user's item sequence
            if user_id in self.user_history:
                user_seq = self.user_history[user_id]
            else:
                # Cold start - return CLIP candidates
                return [(item_id, 0.0) for item_id in candidate_items[:topk]]

            # Convert to internal IDs
            token2id = self.seq_dataset.field2token_id['item_id']
            max_len = self.seq_config['MAX_ITEM_LIST_LENGTH']

            # Build sequence (most recent items)
            item_seq = []
            for item_id in user_seq[-max_len:]:
                if item_id in token2id:
                    item_seq.append(token2id[item_id])

            if not item_seq:
                return [(item_id, 0.0) for item_id in candidate_items[:topk]]

            # Pad sequence
            seq_len = len(item_seq)
            pad_len = max_len - seq_len
            item_seq_padded = [0] * pad_len + item_seq

            # Build interaction for SASRec
            from recbole.data.interaction import Interaction
            interaction = Interaction({
                'item_id_list': torch.tensor([item_seq_padded], device=self.device),
                'item_length': torch.tensor([seq_len], device=self.device),
            })

            # Get scores
            with torch.no_grad():
                scores_all = self.seq_model.full_sort_predict(interaction)
                scores_all = scores_all.view(-1).cpu().numpy()

            # Score candidates
            scored = []
            for item_id in candidate_items:
                if item_id in token2id:
                    internal_id = token2id[item_id]
                    score = scores_all[internal_id]
                    scored.append((item_id, float(score)))

            # Sort by score
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:topk]

        except Exception as e:
            print(f"SASRec ranking failed: {e}")
            import traceback
            traceback.print_exc()
            return [(item_id, 0.0) for item_id in candidate_items[:topk]]

    def get_feed(
        self,
        user_id: str,
        limit: int = 50,
        candidate_pool_size: int = 1000,
        exclude_seen: bool = True
    ) -> List[Dict]:
        """
        Get personalized feed for a user.

        Args:
            user_id: User identifier
            limit: Number of items to return
            candidate_pool_size: Size of CLIP candidate pool
            exclude_seen: Whether to exclude items user has already seen

        Returns:
            List of recommended items with scores
        """
        # Get user's interaction history
        user_items = self.user_history.get(user_id, [])

        # Build user CLIP vector
        if user_items:
            user_vec = self.build_user_clip_vector(user_items)
        else:
            # Cold start - use random popular items or quiz-based
            user_vec = None

        # Get candidates
        exclude = user_items if exclude_seen else None

        if user_vec is not None:
            candidates = self.get_clip_candidates(
                user_vec,
                k=candidate_pool_size,
                exclude_items=exclude
            )
            candidate_ids = [item_id for item_id, _ in candidates]
        else:
            # Cold start fallback: popular items
            candidate_ids = self.item_ids[:candidate_pool_size]

        # Rank with SASRec (or return CLIP order if no model)
        if self.seq_model is not None and user_items:
            ranked = self.rank_with_sasrec(user_id, candidate_ids, topk=limit)
        else:
            # Use CLIP similarity as ranking
            ranked = candidates[:limit] if user_vec is not None else [
                (item_id, 0.0) for item_id in candidate_ids[:limit]
            ]

        # Build response (with final exclusion check)
        seen_set = set(user_items) if exclude_seen else set()
        results = []
        for item_id, score in ranked:
            if item_id not in seen_set:
                results.append({
                    'item_id': item_id,
                    'score': score,
                    'source': 'sasrec' if self.seq_model is not None and user_items else 'clip'
                })

        return results

    def get_similar_items(self, item_id: str, k: int = 10) -> List[Dict]:
        """Get visually similar items using FashionCLIP."""
        if item_id not in self.embeddings_dict:
            return []

        item_vec = self.embeddings_dict[item_id].astype('float32')
        item_vec = item_vec / (np.linalg.norm(item_vec) + 1e-8)

        candidates = self.get_clip_candidates(item_vec, k=k+1, exclude_items=[item_id])

        return [{'item_id': cid, 'similarity': score} for cid, score in candidates[:k]]

    def style_quiz_init(self, liked_items: List[str], disliked_items: List[str] = None) -> np.ndarray:
        """
        Initialize user vector from style quiz responses.

        Args:
            liked_items: Items user liked in quiz
            disliked_items: Items user disliked (used for negative weighting)

        Returns:
            User embedding vector
        """
        liked_vecs = []
        for item_id in liked_items:
            if item_id in self.embeddings_dict:
                liked_vecs.append(self.embeddings_dict[item_id])

        if not liked_vecs:
            return None

        user_vec = np.mean(liked_vecs, axis=0)

        # Optional: subtract disliked items
        if disliked_items:
            disliked_vecs = []
            for item_id in disliked_items:
                if item_id in self.embeddings_dict:
                    disliked_vecs.append(self.embeddings_dict[item_id])

            if disliked_vecs:
                dislike_vec = np.mean(disliked_vecs, axis=0)
                user_vec = user_vec - 0.3 * dislike_vec  # Weighted subtraction

        # Normalize
        user_vec = user_vec / (np.linalg.norm(user_vec) + 1e-8)
        return user_vec.astype('float32')


def test_feed():
    """Test the feed system."""
    os.chdir("/home/ubuntu/recSys/outfitTransformer")

    print("Initializing Amazon Fashion Feed...")
    feed = AmazonFashionFeed(
        duorec_checkpoint="models/sasrec_amazon/SASRec-Dec-12-2025_01-35-54.pth"
    )

    # Test with a random user
    test_users = list(feed.user_history.keys())[:5]

    for user_id in test_users:
        print(f"\n{'='*60}")
        print(f"Feed for user: {user_id}")
        print(f"History: {len(feed.user_history[user_id])} items")

        recommendations = feed.get_feed(user_id, limit=10)

        print(f"\nTop 10 recommendations:")
        for i, item in enumerate(recommendations, 1):
            print(f"  {i}. {item['item_id']} (score: {item['score']:.4f}, source: {item['source']})")

    # Test similar items
    test_item = feed.item_ids[0]
    print(f"\n{'='*60}")
    print(f"Similar items to {test_item}:")
    similar = feed.get_similar_items(test_item, k=5)
    for i, item in enumerate(similar, 1):
        print(f"  {i}. {item['item_id']} (similarity: {item['similarity']:.4f})")


if __name__ == "__main__":
    test_feed()
