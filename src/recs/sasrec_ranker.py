"""
SASRec Ranker Module

Sequential model ranking with OOV median score fallback.

Scoring:
- WARM users (5+ interactions): sasrec=0.40, embedding=0.35, preference=0.25
- COLD/TINDER users: embedding=0.60, preference=0.40

OOV Handling:
- Items not in SASRec vocabulary receive median(all_scores) as fallback
- This allows embedding and preference scores to determine ranking for new items
"""

import os
import sys
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from recs.models import (
    UserState,
    UserStateType,
    Candidate,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SASRecRankerConfig:
    """Configuration for SASRec ranking."""

    # Score weights for WARM users (5+ interactions)
    WARM_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        'sasrec': 0.40,
        'embedding': 0.35,
        'preference': 0.25
    })

    # Score weights for COLD/TINDER users (no SASRec)
    COLD_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        'embedding': 0.60,
        'preference': 0.40
    })

    # Sequence configuration
    MAX_SEQ_LENGTH: int = 50
    MIN_SEQUENCE_FOR_SASREC: int = 5

    # Model paths
    CHECKPOINT_PATH: str = "/home/ubuntu/recSys/outfitTransformer/models/SASRec-Dec-11-2025_18-20-35.pth"
    DATA_PATH: str = "/home/ubuntu/recSys/outfitTransformer/data/amazon_fashion/recbole"
    DATASET_NAME: str = "amazon_mens"


# =============================================================================
# SASRec Ranker
# =============================================================================

class SASRecRanker:
    """
    SASRec sequential model with OOV median score fallback.

    For warm users (5+ interactions):
    - Uses SASRec to predict next-item probabilities
    - OOV items receive median score
    - Combines: sasrec * 0.40 + embedding * 0.35 + preference * 0.25

    For cold/tinder users:
    - No SASRec scoring
    - Combines: embedding * 0.60 + preference * 0.40
    """

    def __init__(self, config: Optional[SASRecRankerConfig] = None, load_model: bool = True):
        """
        Initialize SASRec ranker.

        Args:
            config: Configuration for the ranker
            load_model: Whether to load the SASRec model (can be disabled for testing)
        """
        self.config = config or SASRecRankerConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.dataset = None
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.vocab_size = 0

        if load_model:
            self._load_model()

    def _load_model(self):
        """Load SASRec model and vocabulary from checkpoint."""
        try:
            from recbole.config import Config
            from recbole.data import create_dataset
            from recbole.model.sequential_recommender import SASRec

            config_dict = {
                'data_path': self.config.DATA_PATH,
                'dataset': self.config.DATASET_NAME,
                'USER_ID_FIELD': 'user_id',
                'ITEM_ID_FIELD': 'item_id',
                'TIME_FIELD': 'timestamp',
                'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
                'MAX_ITEM_LIST_LENGTH': self.config.MAX_SEQ_LENGTH,
                'train_neg_sample_args': None,
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

            config = Config(model='SASRec', dataset=self.config.DATASET_NAME, config_dict=config_dict)
            self.dataset = create_dataset(config)

            self.model = SASRec(config, self.dataset).to(self.device)
            state = torch.load(self.config.CHECKPOINT_PATH, map_location=self.device, weights_only=False)

            # Handle vocab size mismatch by resizing embeddings
            state_dict = state['state_dict']
            model_vocab_size = self.dataset.item_num
            checkpoint_vocab_size = state_dict['item_embedding.weight'].shape[0]

            if checkpoint_vocab_size != model_vocab_size:
                print(f"[SASRecRanker] Vocab size mismatch: checkpoint={checkpoint_vocab_size}, model={model_vocab_size}")

                if checkpoint_vocab_size > model_vocab_size:
                    # Truncate checkpoint embeddings to match model
                    print(f"[SASRecRanker] Truncating embeddings from {checkpoint_vocab_size} to {model_vocab_size}")
                    state_dict['item_embedding.weight'] = state_dict['item_embedding.weight'][:model_vocab_size]
                else:
                    # Expand model embeddings (initialize new rows with mean of existing)
                    print(f"[SASRecRanker] Expanding embeddings from {checkpoint_vocab_size} to {model_vocab_size}")
                    old_emb = state_dict['item_embedding.weight']
                    new_emb = torch.zeros(model_vocab_size, old_emb.shape[1], device=old_emb.device)
                    new_emb[:checkpoint_vocab_size] = old_emb
                    # Initialize new embeddings with mean of existing
                    new_emb[checkpoint_vocab_size:] = old_emb.mean(dim=0)
                    state_dict['item_embedding.weight'] = new_emb

            self.model.load_state_dict(state_dict)
            self.model.eval()

            # Build vocabulary mappings
            self.token2id = dict(self.dataset.field2token_id['item_id'])
            self.id2token = {v: k for k, v in self.token2id.items()}
            self.vocab_size = len(self.token2id)

            print(f"[SASRecRanker] Model loaded: {self.vocab_size} items in vocabulary")

        except Exception as e:
            print(f"[SASRecRanker] Warning: Could not load SASRec model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    # =========================================================
    # Main Ranking Method
    # =========================================================

    def rank_candidates(
        self,
        user_state: UserState,
        candidates: List[Candidate]
    ) -> List[Candidate]:
        """
        Rank candidates using weighted score combination.

        For warm users (5+ interactions):
            final_score = 0.40 * sasrec + 0.35 * embedding + 0.25 * preference

        For cold/tinder users:
            final_score = 0.60 * embedding + 0.40 * preference

        Args:
            user_state: User's state with interaction_sequence
            candidates: List of Candidate objects with embedding_score and preference_score set

        Returns:
            Candidates sorted by final_score (descending)
        """
        if not candidates:
            return candidates

        # Determine if SASRec should be used
        use_sasrec = (
            user_state.state_type == UserStateType.WARM_USER and
            self.model is not None and
            len(user_state.interaction_sequence) >= self.config.MIN_SEQUENCE_FOR_SASREC
        )

        # Get SASRec scores (or empty dict for cold users)
        if use_sasrec:
            sasrec_scores = self._get_sasrec_scores(
                user_state.interaction_sequence,
                [c.item_id for c in candidates]
            )
            weights = self.config.WARM_WEIGHTS
        else:
            sasrec_scores = {}
            weights = self.config.COLD_WEIGHTS

        # Normalize all scores to [0, 1]
        emb_scores = [c.embedding_score for c in candidates]
        pref_scores = [c.preference_score for c in candidates]
        sas_scores = [sasrec_scores.get(c.item_id, 0) for c in candidates] if sasrec_scores else [0] * len(candidates)

        emb_norm = self._normalize(emb_scores)
        pref_norm = self._normalize(pref_scores)
        sas_norm = self._normalize(sas_scores) if sasrec_scores else [0] * len(candidates)

        # Compute final scores
        for i, c in enumerate(candidates):
            if 'sasrec' in weights:
                c.final_score = (
                    weights['sasrec'] * sas_norm[i] +
                    weights['embedding'] * emb_norm[i] +
                    weights['preference'] * pref_norm[i]
                )
            else:
                c.final_score = (
                    weights['embedding'] * emb_norm[i] +
                    weights['preference'] * pref_norm[i]
                )

            # Store normalized SASRec score and OOV status
            c.sasrec_score = sasrec_scores.get(c.item_id, 0)
            c.is_oov = c.item_id not in self.token2id if self.model else True

        # Sort by final_score (descending)
        candidates.sort(key=lambda c: c.final_score, reverse=True)
        return candidates

    # =========================================================
    # SASRec Scoring
    # =========================================================

    def _get_sasrec_scores(
        self,
        sequence: List[str],
        candidate_ids: List[str]
    ) -> Dict[str, float]:
        """
        Get SASRec scores for candidates.

        OOV items receive median(all_scores) as fallback.

        Args:
            sequence: User's interaction sequence (item IDs)
            candidate_ids: List of candidate item IDs to score

        Returns:
            Dict mapping item_id -> score
        """
        if not self.model or not sequence:
            return {}

        try:
            # Convert sequence to internal IDs (skip OOV items in sequence)
            item_seq = []
            for item_id in sequence[-self.config.MAX_SEQ_LENGTH:]:
                if item_id in self.token2id:
                    item_seq.append(self.token2id[item_id])

            if not item_seq:
                return {}

            # Pad sequence (left-padding with 0)
            seq_len = len(item_seq)
            pad_len = self.config.MAX_SEQ_LENGTH - seq_len
            seq_padded = [0] * pad_len + item_seq

            # Build interaction for SASRec
            from recbole.data.interaction import Interaction
            interaction = Interaction({
                'item_id_list': torch.tensor([seq_padded], device=self.device),
                'item_length': torch.tensor([seq_len], device=self.device),
            })

            # Get all scores
            with torch.no_grad():
                scores_all = self.model.full_sort_predict(interaction)
                scores_all = scores_all.view(-1).cpu().numpy()

            # Compute median for OOV fallback (exclude padding token at index 0)
            valid_scores = scores_all[1:]  # Skip padding token
            median_score = float(np.median(valid_scores))

            # Score candidates
            result = {}
            for item_id in candidate_ids:
                if item_id in self.token2id:
                    internal_id = self.token2id[item_id]
                    result[item_id] = float(scores_all[internal_id])
                else:
                    # OOV: use median score
                    result[item_id] = median_score

            return result

        except Exception as e:
            print(f"[SASRecRanker] Error in SASRec scoring: {e}")
            import traceback
            traceback.print_exc()
            return {}

    # =========================================================
    # Score Normalization
    # =========================================================

    def _normalize(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range using min-max normalization.

        Args:
            scores: List of raw scores

        Returns:
            List of normalized scores in [0, 1]
        """
        if not scores:
            return scores

        arr = np.array(scores)
        min_val = np.min(arr)
        max_val = np.max(arr)

        if max_val - min_val < 1e-8:
            # All scores are the same, return 0.5 for all
            return [0.5] * len(scores)

        normalized = (arr - min_val) / (max_val - min_val)
        return normalized.tolist()

    # =========================================================
    # Vocabulary Info
    # =========================================================

    def is_in_vocabulary(self, item_id: str) -> bool:
        """Check if an item is in the SASRec vocabulary."""
        return item_id in self.token2id

    def get_vocabulary_coverage(self, item_ids: List[str]) -> Tuple[int, int, float]:
        """
        Get vocabulary coverage for a list of items.

        Returns:
            (in_vocab_count, total_count, coverage_percentage)
        """
        if not self.token2id:
            return 0, len(item_ids), 0.0

        in_vocab = sum(1 for item_id in item_ids if item_id in self.token2id)
        total = len(item_ids)
        coverage = (in_vocab / total * 100) if total > 0 else 0.0

        return in_vocab, total, coverage

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "model_loaded": self.model is not None,
            "vocab_size": self.vocab_size,
            "max_seq_length": self.config.MAX_SEQ_LENGTH,
            "min_seq_for_sasrec": self.config.MIN_SEQUENCE_FOR_SASREC,
            "warm_weights": self.config.WARM_WEIGHTS,
            "cold_weights": self.config.COLD_WEIGHTS,
            "device": str(self.device),
            "checkpoint_path": self.config.CHECKPOINT_PATH
        }


# =============================================================================
# Testing
# =============================================================================

def test_sasrec_ranker():
    """Test the SASRec ranker module."""
    print("=" * 70)
    print("Testing SASRec Ranker Module")
    print("=" * 70)

    # Test 1: Initialize ranker
    print("\n1. Initializing SASRec ranker...")
    ranker = SASRecRanker()
    info = ranker.get_model_info()
    print(f"   Model loaded: {info['model_loaded']}")
    print(f"   Vocab size: {info['vocab_size']}")
    print(f"   Device: {info['device']}")

    # Test 2: Create test candidates
    print("\n2. Creating test candidates...")
    test_candidates = [
        Candidate(
            item_id="B07D5SYZK5",  # May or may not be in vocab
            embedding_score=0.9,
            preference_score=0.7,
            category="tops",
            brand="Nike"
        ),
        Candidate(
            item_id="B08N5WRWNW",
            embedding_score=0.85,
            preference_score=0.8,
            category="tops",
            brand="Adidas"
        ),
        Candidate(
            item_id="test_oov_item",  # Definitely not in vocab
            embedding_score=0.95,
            preference_score=0.6,
            category="tops",
            brand="Unknown"
        ),
    ]
    print(f"   Created {len(test_candidates)} test candidates")

    # Test 3: Rank candidates for COLD user
    print("\n3. Ranking for COLD user (no SASRec)...")
    cold_state = UserState(
        user_id="test_cold_user",
        state_type=UserStateType.COLD_START,
        interaction_sequence=[]
    )

    ranked_cold = ranker.rank_candidates(cold_state, test_candidates.copy())
    print(f"   Cold ranking (embedding * 0.60 + preference * 0.40):")
    for i, c in enumerate(ranked_cold[:3]):
        print(f"   {i+1}. {c.item_id[:15]}... final={c.final_score:.3f} (emb={c.embedding_score:.2f}, pref={c.preference_score:.2f})")

    # Test 4: Check vocabulary coverage
    print("\n4. Checking vocabulary coverage...")
    item_ids = [c.item_id for c in test_candidates]
    in_vocab, total, coverage = ranker.get_vocabulary_coverage(item_ids)
    print(f"   In vocab: {in_vocab}/{total} ({coverage:.1f}%)")

    # Test 5: Rank candidates for WARM user (if model loaded)
    if ranker.model:
        print("\n5. Ranking for WARM user (with SASRec)...")

        # Use some known items from the vocab if available
        sample_sequence = list(ranker.token2id.keys())[:10] if ranker.token2id else []

        if sample_sequence:
            warm_state = UserState(
                user_id="test_warm_user",
                state_type=UserStateType.WARM_USER,
                interaction_sequence=sample_sequence
            )

            # Create candidates with some vocab items
            warm_candidates = [
                Candidate(
                    item_id=sample_sequence[0] if len(sample_sequence) > 0 else "item1",
                    embedding_score=0.9,
                    preference_score=0.7,
                    category="tops",
                    brand="Nike"
                ),
                Candidate(
                    item_id=sample_sequence[1] if len(sample_sequence) > 1 else "item2",
                    embedding_score=0.85,
                    preference_score=0.8,
                    category="tops",
                    brand="Adidas"
                ),
                Candidate(
                    item_id="oov_item_xyz",  # OOV
                    embedding_score=0.95,
                    preference_score=0.6,
                    category="tops",
                    brand="Unknown"
                ),
            ]

            ranked_warm = ranker.rank_candidates(warm_state, warm_candidates.copy())
            print(f"   Warm ranking (sasrec * 0.40 + embedding * 0.35 + preference * 0.25):")
            for i, c in enumerate(ranked_warm[:3]):
                oov_str = "(OOV)" if c.is_oov else ""
                print(f"   {i+1}. {c.item_id[:15]}... final={c.final_score:.3f} sas={c.sasrec_score:.3f} {oov_str}")
        else:
            print("   (Skipped - no items in vocabulary)")

    print("\n" + "=" * 70)
    print("SASRec Ranker test complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_sasrec_ranker()
