"""
Hybrid Feed Generator Module
Combines visual similarity (FashionCLIP) with sequential recommendations (BERT4Rec)
"""
import os
import random
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

from embeddings import FaissSearcher, FashionEmbeddingGenerator


class HybridFeedGenerator:
    """
    Fashion feed generator combining visual similarity and collaborative filtering
    """

    def __init__(
        self,
        embeddings_path: str,
        faiss_path: Optional[str] = None,
        recbole_checkpoint: Optional[str] = None,
        item_metadata_path: Optional[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize feed generator

        Args:
            embeddings_path: Path to FashionCLIP embeddings
            faiss_path: Path to Faiss index (optional, will build if not provided)
            recbole_checkpoint: Path to RecBole model checkpoint
            item_metadata_path: Path to item metadata JSON
            use_gpu: Whether to use GPU
        """
        # Load visual search
        self.visual_searcher = FaissSearcher(embeddings_path, faiss_path, use_gpu)
        self.item_ids = self.visual_searcher.item_ids
        self.embeddings = self.visual_searcher.embeddings

        # Load BERT4Rec model if provided
        self.bert4rec_model = None
        self.bert4rec_config = None
        self.bert4rec_dataset = None
        self.bert4rec_checkpoint_path = recbole_checkpoint
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

        if recbole_checkpoint and os.path.exists(recbole_checkpoint):
            self._load_bert4rec_model(recbole_checkpoint)

        # Load item metadata if provided
        self.item_metadata = {}
        if item_metadata_path and os.path.exists(item_metadata_path):
            import json
            with open(item_metadata_path) as f:
                self.item_metadata = json.load(f)

        # Build category index
        self.category_to_items = self._build_category_index()

        print(f"HybridFeedGenerator ready:")
        print(f"  Items: {len(self.item_ids)}")
        print(f"  BERT4Rec: {'Loaded' if self.bert4rec_model else 'Not loaded'}")
        print(f"  Device: {self.device}")
        print(f"  Categories: {len(self.category_to_items)}")

    def _load_bert4rec_model(self, checkpoint_path: str):
        """Load BERT4Rec model for sequential recommendations"""
        try:
            from recbole.quick_start import load_data_and_model
            import torch
            from pathlib import Path

            # RecBole uses relative paths from training, so we need to cd to project root
            # Checkpoint is typically at project_root/models/BERT4Rec-*.pth
            checkpoint_path = str(Path(checkpoint_path).resolve())
            project_root = str(Path(checkpoint_path).parent.parent)  # models/../ = project root

            original_cwd = os.getcwd()

            # PyTorch 2.6+ changed default weights_only=True, but RecBole doesn't handle this
            # Temporarily patch torch.load to use weights_only=False for trusted checkpoints
            original_torch_load = torch.load
            def patched_load(*args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)

            torch.load = patched_load
            try:
                # Change to project root for RecBole to find dataset
                os.chdir(project_root)
                config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
                    checkpoint_path
                )
            finally:
                # Restore original torch.load and working directory
                torch.load = original_torch_load
                os.chdir(original_cwd)

            self.bert4rec_config = config
            self.bert4rec_model = model.to(self.device)
            self.bert4rec_model.eval()
            self.bert4rec_dataset = dataset

            # Build item ID mapping (RecBole internal ID -> original item ID)
            self.bert4rec_id2token = dataset.field2id_token['item_id']
            self.bert4rec_token2id = dataset.field2token_id['item_id']

            print(f"Loaded BERT4Rec model from {checkpoint_path}")
            print(f"  Items in model: {dataset.item_num}")
            print(f"  Max sequence length: {config['MAX_ITEM_LIST_LENGTH']}")

        except Exception as e:
            print(f"Warning: Could not load BERT4Rec model: {e}")
            import traceback
            traceback.print_exc()
            self.bert4rec_model = None

    def _build_category_index(self) -> Dict[str, List[str]]:
        """Build index of items by category"""
        category_to_items = defaultdict(list)

        for item_id in self.item_ids:
            if item_id in self.item_metadata:
                category = (
                    self.item_metadata[item_id].get('category_name') or
                    self.item_metadata[item_id].get('semantic_category') or
                    self.item_metadata[item_id].get('category_id') or
                    self.item_metadata[item_id].get('category') or
                    'unknown'
                )
                category_to_items[category].append(item_id)
            else:
                category_to_items['unknown'].append(item_id)

        return dict(category_to_items)

    def get_visual_similar(
        self,
        item_id: str,
        k: int = 20,
        exclude_ids: Optional[Set[str]] = None
    ) -> List[Dict]:
        """
        Get visually similar items

        Args:
            item_id: Query item ID
            k: Number of results
            exclude_ids: Set of item IDs to exclude

        Returns:
            List of dicts with 'item_id' and 'score'
        """
        if item_id not in self.embeddings:
            return []

        # Get more results to account for exclusions and duplicates
        search_k = (k + (len(exclude_ids) if exclude_ids else 0) + 1) * 3
        results = self.visual_searcher.search(item_id, k=search_k)

        # Deduplicate results (keep first occurrence with highest score)
        seen_ids = set()
        unique_results = []
        for r in results:
            if r['item_id'] not in seen_ids:
                seen_ids.add(r['item_id'])
                unique_results.append(r)

        # Filter exclusions
        if exclude_ids:
            unique_results = [r for r in unique_results if r['item_id'] not in exclude_ids]

        return unique_results[:k]

    def get_sequential_recommendations(
        self,
        item_sequence: List[str],
        k: int = 20,
        exclude_ids: Optional[Set[str]] = None
    ) -> List[Dict]:
        """
        Get BERT4Rec sequential recommendations based on item history

        Uses masked prediction: given a sequence of items, predict what comes next.

        Args:
            item_sequence: List of item IDs (user's recent history)
            k: Number of recommendations
            exclude_ids: Set of item IDs to exclude

        Returns:
            List of dicts with 'item_id' and 'score'
        """
        if self.bert4rec_model is None:
            return []

        try:
            # Convert item IDs to RecBole internal IDs
            max_len = self.bert4rec_config['MAX_ITEM_LIST_LENGTH']

            # Filter to items that exist in BERT4Rec vocabulary
            valid_items = []
            for item_id in item_sequence:
                if str(item_id) in self.bert4rec_token2id:
                    valid_items.append(self.bert4rec_token2id[str(item_id)])

            if not valid_items:
                return []

            # Truncate to max length (keep most recent)
            if len(valid_items) > max_len - 1:  # -1 for mask token
                valid_items = valid_items[-(max_len - 1):]

            # Create input sequence with mask at the end
            # BERT4Rec uses mask_token at position to predict
            item_seq = torch.zeros(max_len, dtype=torch.long, device=self.device)
            item_seq_len = len(valid_items) + 1  # +1 for mask

            # Fill sequence (left padding, sequence ends with mask)
            start_idx = max_len - item_seq_len
            for i, item_id in enumerate(valid_items):
                item_seq[start_idx + i] = item_id

            # Add mask token at the end (RecBole uses item_num as mask token)
            mask_token = self.bert4rec_dataset.item_num
            item_seq[max_len - 1] = mask_token

            # Create batch
            item_seq = item_seq.unsqueeze(0)  # [1, max_len]
            item_seq_len = torch.tensor([item_seq_len], device=self.device)

            # Forward pass
            with torch.no_grad():
                # BERT4Rec forward expects interaction dict
                interaction = {
                    'item_id_list': item_seq,
                    'item_length': item_seq_len,
                }

                # Get predictions for the masked position
                output = self.bert4rec_model.forward(item_seq, item_seq_len)

                # Output shape: [batch, seq_len, item_num]
                # Get prediction for last position (masked position)
                scores = output[0, -1, :].cpu().numpy()

            # Build results
            exclude_ids = exclude_ids or set()
            results = []

            # Sort by score descending
            sorted_indices = scores.argsort()[::-1]

            for idx in sorted_indices:
                if idx == 0:  # Skip padding token
                    continue
                if idx >= len(self.bert4rec_id2token):
                    continue

                item_token = self.bert4rec_id2token[idx]
                if item_token == '[PAD]':
                    continue

                if str(item_token) in exclude_ids:
                    continue

                results.append({
                    'item_id': str(item_token),
                    'score': float(scores[idx])
                })

                if len(results) >= k:
                    break

            return results

        except Exception as e:
            print(f"BERT4Rec recommendation error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def generate_feed(
        self,
        user_id: str,
        history: List[str],
        k: int = 20,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """
        Generate personalized feed using hybrid scoring

        Combines:
        - Visual similarity (FashionCLIP): Find items visually similar to history
        - Sequential prediction (BERT4Rec): Predict next items based on sequence
        - Diversity bonus: Encourage category diversity

        Args:
            user_id: User identifier
            history: List of item IDs user has interacted with
            k: Number of recommendations
            weights: Weight configuration:
                - visual: Weight for visual similarity (default: 0.4)
                - sequential: Weight for BERT4Rec predictions (default: 0.4)
                - diversity: Diversity bonus (default: 0.2)

        Returns:
            List of recommended items with scores
        """
        if weights is None:
            weights = {'visual': 0.4, 'sequential': 0.4, 'diversity': 0.2}

        history_set = set(str(h) for h in history)
        candidates = defaultdict(float)
        candidate_sources = defaultdict(list)

        # 1. Visual similarity from history (FashionCLIP)
        if weights.get('visual', 0) > 0 and history:
            recent_history = history[-10:]  # Use last 10 items
            for hist_item in recent_history:
                similar = self.get_visual_similar(str(hist_item), k=20, exclude_ids=history_set)
                for item in similar:
                    item_id = str(item['item_id'])
                    if item_id not in history_set:
                        # Normalize visual scores (cosine similarity typically 0-1)
                        norm_score = min(1.0, max(0.0, item['score']))
                        score = norm_score * weights['visual']
                        candidates[item_id] += score
                        candidate_sources[item_id].append('visual')

        # 2. Sequential recommendations (BERT4Rec)
        if weights.get('sequential', 0) > 0 and history and self.bert4rec_model is not None:
            seq_recs = self.get_sequential_recommendations(
                history[-50:],  # Use up to 50 recent items
                k=100,
                exclude_ids=history_set
            )
            if seq_recs:
                # Normalize BERT4Rec scores using softmax-style normalization
                max_score = max(r['score'] for r in seq_recs) if seq_recs else 1.0
                min_score = min(r['score'] for r in seq_recs) if seq_recs else 0.0
                score_range = max_score - min_score if max_score != min_score else 1.0

                for item in seq_recs:
                    item_id = str(item['item_id'])
                    if item_id not in history_set:
                        # Normalize to 0-1 range
                        norm_score = (item['score'] - min_score) / score_range
                        score = norm_score * weights['sequential']
                        candidates[item_id] += score
                        candidate_sources[item_id].append('sequential')

        # 3. Diversity bonus for items from different categories
        if weights.get('diversity', 0) > 0:
            history_categories = set()
            for item_id in history:
                if str(item_id) in self.item_metadata:
                    cat = self.item_metadata[str(item_id)].get('category_name') or \
                          self.item_metadata[str(item_id)].get('semantic_category', 'unknown')
                    history_categories.add(cat)

            for item_id in list(candidates.keys()):
                if item_id in self.item_metadata:
                    cat = self.item_metadata[item_id].get('category_name') or \
                          self.item_metadata[item_id].get('semantic_category', 'unknown')
                    if cat not in history_categories:
                        candidates[item_id] += weights['diversity']
                        candidate_sources[item_id].append('diversity')

        # If no candidates, return popular/random items
        if not candidates:
            return self._get_cold_start_items(k, history_set)

        # Rank and return top-k
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:k]

        return [
            {
                'item_id': item_id,
                'score': score,
                'sources': candidate_sources[item_id]
            }
            for item_id, score in ranked
        ]

    def _get_cold_start_items(
        self,
        k: int,
        exclude_ids: Set[str]
    ) -> List[Dict]:
        """
        Get items for users with no history (cold start)

        Args:
            k: Number of items
            exclude_ids: Items to exclude

        Returns:
            List of items
        """
        # Return random items from each category
        available_items = [iid for iid in self.item_ids if iid not in exclude_ids]

        if len(available_items) <= k:
            selected = available_items
        else:
            selected = random.sample(available_items, k)

        return [
            {'item_id': item_id, 'score': 0.5, 'sources': ['cold_start']}
            for item_id in selected
        ]

    def apply_diversity_reranking(
        self,
        candidates: List[Dict],
        k: int,
        lambda_param: float = 0.5
    ) -> List[Dict]:
        """
        Apply MMR-style diversity reranking

        Args:
            candidates: List of candidate items with scores
            k: Number of items to return
            lambda_param: Balance between relevance and diversity (0-1)

        Returns:
            Reranked list of items
        """
        if len(candidates) <= k:
            return candidates

        selected = []
        remaining = list(candidates)

        while len(selected) < k and remaining:
            best_idx = 0
            best_score = float('-inf')

            for i, candidate in enumerate(remaining):
                relevance = candidate['score']

                # Compute max similarity to already selected
                max_sim = 0
                if selected:
                    cand_emb = self.embeddings.get(candidate['item_id'])
                    if cand_emb is not None:
                        for sel in selected:
                            sel_emb = self.embeddings.get(sel['item_id'])
                            if sel_emb is not None:
                                sim = np.dot(cand_emb, sel_emb)
                                max_sim = max(max_sim, sim)

                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    def get_item_details(self, item_id: str) -> Optional[Dict]:
        """
        Get item metadata

        Args:
            item_id: Item ID

        Returns:
            Item metadata or None
        """
        if item_id in self.item_metadata:
            return {
                'item_id': item_id,
                **self.item_metadata[item_id]
            }
        elif item_id in self.embeddings:
            return {'item_id': item_id, 'has_embedding': True}
        return None

    def get_category_items(
        self,
        category: str,
        k: int = 20,
        exclude_ids: Optional[Set[str]] = None
    ) -> List[str]:
        """
        Get items from a specific category

        Args:
            category: Category name
            k: Number of items
            exclude_ids: Items to exclude

        Returns:
            List of item IDs
        """
        items = self.category_to_items.get(category, [])

        if exclude_ids:
            items = [iid for iid in items if iid not in exclude_ids]

        if len(items) <= k:
            return items

        return random.sample(items, k)


def create_feed_generator(
    embeddings_path: str = "models/polyvore_embeddings.pkl",
    faiss_path: str = "models/polyvore_faiss_index.bin",
    recbole_checkpoint: Optional[str] = None,
    metadata_path: Optional[str] = "data/polyvore/polyvore_item_metadata.json",
    use_gpu: bool = True
) -> HybridFeedGenerator:
    """
    Factory function to create feed generator

    Args:
        embeddings_path: Path to embeddings
        faiss_path: Path to Faiss index
        recbole_checkpoint: Path to RecBole model
        metadata_path: Path to item metadata
        use_gpu: Whether to use GPU

    Returns:
        Initialized HybridFeedGenerator
    """
    return HybridFeedGenerator(
        embeddings_path=embeddings_path,
        faiss_path=faiss_path,
        recbole_checkpoint=recbole_checkpoint,
        item_metadata_path=metadata_path,
        use_gpu=use_gpu
    )


if __name__ == "__main__":
    # Test the feed generator
    generator = create_feed_generator(use_gpu=False)

    # Test with sample history
    if generator.item_ids:
        test_history = generator.item_ids[:5]
        print(f"\nTest history: {test_history}")

        feed = generator.generate_feed(
            user_id="test_user",
            history=test_history,
            k=10
        )

        print("\nGenerated feed:")
        for i, item in enumerate(feed, 1):
            print(f"{i}. {item['item_id']} (score: {item['score']:.4f}, sources: {item.get('sources', [])})")
