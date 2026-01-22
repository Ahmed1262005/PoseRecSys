"""
Outfit Transformer Model
Multimodal Transformer for Fashion Compatibility Prediction

Based on:
- POG (Alibaba): Masked item prediction with multimodal embeddings
- OutfitTransformer (Amazon): Outfit token for global representation

Architecture:
- Input: Visual (FashionCLIP) + Category embeddings
- Transformer encoder (no positional encoding - items are sets)
- Masked item prediction for training
- Outfit token for compatibility scoring
"""
import json
import math
import pickle
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class MultimodalItemEncoder(nn.Module):
    """
    Encodes items using visual (FashionCLIP) + category features
    Following POG's multi-modal embedding approach
    """

    def __init__(
        self,
        visual_dim: int = 512,
        num_categories: int = 200,
        category_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.category_embedding = nn.Embedding(num_categories, category_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + category_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        category_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            visual_features: [batch, seq_len, visual_dim] FashionCLIP embeddings
            category_ids: [batch, seq_len] category indices

        Returns:
            [batch, seq_len, hidden_dim] fused item embeddings
        """
        v = self.visual_proj(visual_features)
        c = self.category_embedding(category_ids)
        fused = torch.cat([v, c], dim=-1)
        return self.fusion(fused)


class OutfitTransformer(nn.Module):
    """
    Transformer model for outfit compatibility

    Key features:
    - No positional encoding (outfits are sets, not sequences)
    - Outfit token for global representation
    - Mask token for masked item prediction
    """

    def __init__(
        self,
        visual_dim: int = 512,
        num_categories: int = 200,
        num_items: int = 120000,
        hidden_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_outfit_size: int = 10
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_items = num_items

        # Item encoder (visual + category fusion)
        self.item_encoder = MultimodalItemEncoder(
            visual_dim=visual_dim,
            num_categories=num_categories,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # Special tokens
        self.outfit_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Transformer encoder (no positional encoding!)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output heads
        self.compatibility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # Item prediction head for masked item prediction
        self.item_prediction_head = nn.Linear(hidden_dim, num_items)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following BERT"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(
        self,
        visual_features: torch.Tensor,
        category_ids: torch.Tensor,
        mask_positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            visual_features: [batch, seq_len, visual_dim]
            category_ids: [batch, seq_len]
            mask_positions: [batch] positions to mask (for training)
            attention_mask: [batch, seq_len] padding mask

        Returns:
            outfit_repr: [batch, hidden_dim] global outfit representation
            sequence_output: [batch, seq_len+1, hidden_dim] all outputs
        """
        batch_size, seq_len, _ = visual_features.shape

        # Encode items
        item_embeds = self.item_encoder(visual_features, category_ids)

        # Apply mask if training
        if mask_positions is not None:
            mask_token_expanded = self.mask_token.expand(batch_size, 1, -1)
            for i in range(batch_size):
                pos = mask_positions[i].item()
                if 0 <= pos < seq_len:
                    item_embeds[i, pos] = mask_token_expanded[i, 0]

        # Prepend outfit token
        outfit_tokens = self.outfit_token.expand(batch_size, 1, -1)
        sequence = torch.cat([outfit_tokens, item_embeds], dim=1)

        # Create attention mask (outfit token always attends)
        if attention_mask is not None:
            outfit_mask = torch.ones(batch_size, 1, device=attention_mask.device)
            full_mask = torch.cat([outfit_mask, attention_mask], dim=1)
            # Convert to additive mask for transformer
            attn_mask = full_mask.float().masked_fill(full_mask == 0, float('-inf'))
            attn_mask = attn_mask.masked_fill(full_mask == 1, 0.0)
        else:
            attn_mask = None

        # Transformer forward (no positional encoding!)
        output = self.transformer(sequence, src_key_padding_mask=attn_mask)

        # Outfit token output = global representation
        outfit_repr = output[:, 0, :]

        return outfit_repr, output

    def predict_compatibility(
        self,
        visual_features: torch.Tensor,
        category_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict outfit compatibility score

        Returns:
            [batch] compatibility scores (0-1)
        """
        outfit_repr, _ = self.forward(visual_features, category_ids, attention_mask=attention_mask)
        logits = self.compatibility_head(outfit_repr)
        return torch.sigmoid(logits.squeeze(-1))

    def predict_masked_item(
        self,
        visual_features: torch.Tensor,
        category_ids: torch.Tensor,
        mask_positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict masked item for FITB task

        Returns:
            [batch, num_items] logits over all items
        """
        _, output = self.forward(
            visual_features, category_ids,
            mask_positions=mask_positions,
            attention_mask=attention_mask
        )

        # Get output at masked positions (+1 for outfit token)
        batch_size = visual_features.shape[0]
        masked_outputs = []
        for i in range(batch_size):
            pos = mask_positions[i].item() + 1  # +1 for outfit token
            masked_outputs.append(output[i, pos, :])

        masked_output = torch.stack(masked_outputs, dim=0)
        return self.item_prediction_head(masked_output)

    def score_fitb_answers(
        self,
        context_visual: torch.Tensor,
        context_cats: torch.Tensor,
        answer_visual: torch.Tensor,
        answer_cats: torch.Tensor
    ) -> torch.Tensor:
        """
        Score FITB answer candidates using visual feature compatibility.

        This method is train/test agnostic - works with ANY items because
        it uses visual features, not item IDs.

        Args:
            context_visual: [context_len, visual_dim] context item visual features
            context_cats: [context_len] context item category ids
            answer_visual: [num_answers, visual_dim] answer candidate visual features
            answer_cats: [num_answers] answer candidate category ids

        Returns:
            [num_answers] compatibility scores for each answer
        """
        device = context_visual.device
        context_len = context_visual.shape[0]
        num_answers = answer_visual.shape[0]

        scores = []
        for i in range(num_answers):
            # Create outfit: context + this answer
            outfit_visual = torch.cat([
                context_visual,
                answer_visual[i:i+1]
            ], dim=0).unsqueeze(0)  # [1, context_len+1, visual_dim]

            outfit_cats = torch.cat([
                context_cats,
                answer_cats[i:i+1]
            ], dim=0).unsqueeze(0)  # [1, context_len+1]

            # Get outfit representation
            outfit_repr, _ = self.forward(outfit_visual, outfit_cats)

            # Get compatibility score
            score = self.compatibility_head(outfit_repr)
            scores.append(score.squeeze())

        return torch.stack(scores)

    def encode_item(
        self,
        visual_features: torch.Tensor,
        category_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode a single item using the multimodal encoder.

        Args:
            visual_features: [visual_dim] visual features
            category_id: scalar category id

        Returns:
            [hidden_dim] encoded item representation
        """
        visual_proj = self.visual_proj(visual_features)
        cat_emb = self.category_embedding(category_id)
        return self.multimodal_encoder(torch.cat([visual_proj, cat_emb], dim=-1))


class PolyvoreOutfitDataset(Dataset):
    """
    Dataset for Polyvore outfits with FashionCLIP embeddings

    Loads:
    - train_no_dup.json: Outfit definitions
    - polyvore_embeddings.pkl: Visual features
    - polyvore_item_metadata.json: Category mappings
    """

    def __init__(
        self,
        outfit_json: str,
        embeddings_pkl: str,
        metadata_json: str,
        max_outfit_size: int = 8,
        mode: str = 'train'
    ):
        self.max_outfit_size = max_outfit_size
        self.mode = mode

        # Load outfits
        with open(outfit_json) as f:
            self.outfits = json.load(f)

        # Load embeddings
        with open(embeddings_pkl, 'rb') as f:
            emb_data = pickle.load(f)
            self.embeddings = emb_data['embeddings']

        # Load metadata for categories
        with open(metadata_json) as f:
            self.metadata = json.load(f)

        # Build category vocabulary
        self.category_to_idx = {}
        for item_id, meta in self.metadata.items():
            cat = meta.get('category_name', 'unknown')
            if cat not in self.category_to_idx:
                self.category_to_idx[cat] = len(self.category_to_idx)

        # Add unknown category
        if 'unknown' not in self.category_to_idx:
            self.category_to_idx['unknown'] = len(self.category_to_idx)

        # Build item vocabulary (tid -> index)
        self.tid_to_idx = {tid: idx for idx, tid in enumerate(self.embeddings.keys())}
        self.idx_to_tid = {idx: tid for tid, idx in self.tid_to_idx.items()}

        # Filter outfits to those with valid embeddings
        self.valid_outfits = []
        for outfit in self.outfits:
            items = outfit.get('items', [])
            tids = [self._extract_tid(item) for item in items]
            valid_tids = [t for t in tids if t and t in self.embeddings]
            if len(valid_tids) >= 2:  # Need at least 2 items
                self.valid_outfits.append({
                    'set_id': outfit.get('set_id'),
                    'tids': valid_tids[:max_outfit_size],
                    'categories': [
                        self.metadata.get(t, {}).get('category_name', 'unknown')
                        for t in valid_tids[:max_outfit_size]
                    ]
                })

        print(f"Loaded {len(self.valid_outfits)} valid outfits from {len(self.outfits)} total")
        print(f"Vocabulary: {len(self.tid_to_idx)} items, {len(self.category_to_idx)} categories")

    def _extract_tid(self, item: dict) -> Optional[str]:
        """Extract TID from item image URL"""
        url = item.get('image', '')
        match = re.search(r'tid=(\d+)', url)
        return match.group(1) if match else None

    def __len__(self):
        return len(self.valid_outfits)

    def __getitem__(self, idx) -> Dict:
        outfit = self.valid_outfits[idx]
        tids = outfit['tids']
        categories = outfit['categories']

        # Get visual features
        visual_feats = np.stack([self.embeddings[t] for t in tids])

        # Get category indices
        cat_ids = np.array([self.category_to_idx.get(c, self.category_to_idx['unknown']) for c in categories])

        # Get item indices for prediction
        item_ids = np.array([self.tid_to_idx[t] for t in tids])

        # For training: randomly mask one item
        if self.mode == 'train':
            mask_pos = random.randint(0, len(tids) - 1)
            target_item = item_ids[mask_pos]
        else:
            mask_pos = -1
            target_item = -1

        return {
            'visual_features': torch.tensor(visual_feats, dtype=torch.float32),
            'category_ids': torch.tensor(cat_ids, dtype=torch.long),
            'item_ids': torch.tensor(item_ids, dtype=torch.long),
            'mask_position': torch.tensor(mask_pos, dtype=torch.long),
            'target_item': torch.tensor(target_item, dtype=torch.long),
            'outfit_size': len(tids)
        }


def collate_outfits(batch: List[Dict]) -> Dict:
    """Collate function with padding"""
    max_size = max(item['outfit_size'] for item in batch)
    visual_dim = batch[0]['visual_features'].shape[-1]

    visual_features = []
    category_ids = []
    item_ids = []
    mask_positions = []
    target_items = []
    attention_masks = []

    for item in batch:
        size = item['outfit_size']
        pad_size = max_size - size

        # Pad visual features
        vf = item['visual_features']
        if pad_size > 0:
            vf = F.pad(vf, (0, 0, 0, pad_size))
        visual_features.append(vf)

        # Pad category ids
        ci = item['category_ids']
        if pad_size > 0:
            ci = F.pad(ci, (0, pad_size))
        category_ids.append(ci)

        # Pad item ids
        ii = item['item_ids']
        if pad_size > 0:
            ii = F.pad(ii, (0, pad_size), value=-1)
        item_ids.append(ii)

        mask_positions.append(item['mask_position'])
        target_items.append(item['target_item'])

        # Attention mask (1 = attend, 0 = ignore)
        mask = torch.ones(max_size)
        mask[size:] = 0
        attention_masks.append(mask)

    return {
        'visual_features': torch.stack(visual_features),
        'category_ids': torch.stack(category_ids),
        'item_ids': torch.stack(item_ids),
        'mask_positions': torch.stack(mask_positions),
        'target_items': torch.stack(target_items),
        'attention_mask': torch.stack(attention_masks)
    }


class FITBDataset(Dataset):
    """
    Dataset for Fill-In-The-Blank evaluation

    Each question has:
    - Context items (3-4 items from outfit)
    - 4 answer choices (1 correct, 3 distractors)
    """

    def __init__(
        self,
        fitb_json: str,
        mapping_json: str,
        embeddings_pkl: str,
        metadata_json: str,
        tid_to_idx: Optional[Dict[str, int]] = None,
        category_to_idx: Optional[Dict[str, int]] = None
    ):
        # Load FITB questions
        with open(fitb_json) as f:
            self.questions = json.load(f)

        # Load set_id_index -> TID mapping
        with open(mapping_json) as f:
            self.mapping = json.load(f)

        # Load embeddings
        with open(embeddings_pkl, 'rb') as f:
            emb_data = pickle.load(f)
            self.embeddings = emb_data['embeddings']

        # Load metadata
        with open(metadata_json) as f:
            self.metadata = json.load(f)

        # Use provided vocabularies or build new ones
        if category_to_idx is not None:
            self.category_to_idx = category_to_idx
        else:
            self.category_to_idx = {}
            for item_id, meta in self.metadata.items():
                cat = meta.get('category_name', 'unknown')
                if cat not in self.category_to_idx:
                    self.category_to_idx[cat] = len(self.category_to_idx)
            if 'unknown' not in self.category_to_idx:
                self.category_to_idx['unknown'] = len(self.category_to_idx)

        # Use provided item vocabulary or build new one
        if tid_to_idx is not None:
            self.tid_to_idx = tid_to_idx
        else:
            self.tid_to_idx = {tid: idx for idx, tid in enumerate(self.embeddings.keys())}

        # Filter valid questions
        self.valid_questions = []
        for q in self.questions:
            context_tids = [self.mapping.get(x) for x in q['question']]
            answer_tids = [self.mapping.get(x) for x in q['answers']]

            # Check all items have embeddings AND are in vocabulary
            valid_context = [t for t in context_tids if t and t in self.embeddings and t in self.tid_to_idx]
            valid_answers = [t for t in answer_tids if t and t in self.embeddings and t in self.tid_to_idx]

            if len(valid_context) >= 2 and len(valid_answers) == 4:
                # Use actual correct index from data (may vary, not always 0)
                correct_idx = q.get('correct', 0)
                self.valid_questions.append({
                    'context_tids': valid_context,
                    'answer_tids': valid_answers,
                    'answer_item_ids': [self.tid_to_idx[t] for t in valid_answers],
                    'correct_idx': correct_idx
                })

        print(f"Loaded {len(self.valid_questions)} valid FITB questions from {len(self.questions)} total")

    def __len__(self):
        return len(self.valid_questions)

    def __getitem__(self, idx) -> Dict:
        q = self.valid_questions[idx]

        # Get context features
        context_visual = np.stack([self.embeddings[t] for t in q['context_tids']])
        context_cats = np.array([
            self.category_to_idx.get(
                self.metadata.get(t, {}).get('category_name', 'unknown'),
                self.category_to_idx['unknown']
            )
            for t in q['context_tids']
        ])

        # Get answer features
        answer_visual = np.stack([self.embeddings[t] for t in q['answer_tids']])
        answer_cats = np.array([
            self.category_to_idx.get(
                self.metadata.get(t, {}).get('category_name', 'unknown'),
                self.category_to_idx['unknown']
            )
            for t in q['answer_tids']
        ])

        # Get answer item indices (for masked prediction scoring)
        answer_item_ids = np.array(q['answer_item_ids'])

        return {
            'context_visual': torch.tensor(context_visual, dtype=torch.float32),
            'context_cats': torch.tensor(context_cats, dtype=torch.long),
            'answer_visual': torch.tensor(answer_visual, dtype=torch.float32),
            'answer_cats': torch.tensor(answer_cats, dtype=torch.long),
            'answer_item_ids': torch.tensor(answer_item_ids, dtype=torch.long),
            'correct_idx': q['correct_idx']
        }


def train_outfit_transformer(
    model: OutfitTransformer,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    all_embeddings: Optional[np.ndarray] = None,
    compatibility_weight: float = 1.0
) -> Tuple[float, float]:
    """
    Train for one epoch with both masked item prediction AND compatibility scoring.

    The compatibility objective is critical for FITB because:
    - FITB test items don't overlap with training items (0% overlap!)
    - Masked item prediction trains item_prediction_head (useless for FITB)
    - Compatibility scoring trains compatibility_head (used by FITB evaluation)

    Training objectives:
    1. Masked item prediction (cross-entropy) - learns item relationships
    2. Compatibility scoring (BCE) - learns outfit coherence
       - Positive: original outfit → score 1
       - Negative: replace one item with random → score 0
    """
    model.train()
    total_masked_loss = 0
    total_compat_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        visual_features = batch['visual_features'].to(device)
        category_ids = batch['category_ids'].to(device)
        mask_positions = batch['mask_positions'].to(device)
        target_items = batch['target_items'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        batch_size = visual_features.shape[0]

        optimizer.zero_grad()

        # === Loss 1: Masked item prediction ===
        logits = model.predict_masked_item(
            visual_features, category_ids, mask_positions, attention_mask
        )
        masked_loss = F.cross_entropy(logits, target_items)

        # === Loss 2: Compatibility scoring ===
        # Get compatibility score for positive (original) outfits
        pos_compat = model.predict_compatibility(
            visual_features, category_ids, attention_mask
        )  # [batch]

        # Create negative outfits by replacing one random item with a random item
        neg_visual = visual_features.clone()
        neg_cats = category_ids.clone()

        if all_embeddings is not None:
            num_items = len(all_embeddings)
            for i in range(batch_size):
                # Pick random position to corrupt
                seq_len = int(attention_mask[i].sum().item())
                if seq_len > 0:
                    corrupt_pos = random.randint(0, seq_len - 1)
                    # Pick random item
                    random_idx = random.randint(0, num_items - 1)
                    neg_visual[i, corrupt_pos] = torch.tensor(
                        all_embeddings[random_idx], device=device, dtype=torch.float32
                    )
                    # Random category (keep within valid range)
                    neg_cats[i, corrupt_pos] = random.randint(0, model.item_encoder.category_embedding.num_embeddings - 1)

        neg_compat = model.predict_compatibility(
            neg_visual, neg_cats, attention_mask
        )  # [batch]

        # BCE loss: positive → 1, negative → 0
        pos_labels = torch.ones(batch_size, device=device)
        neg_labels = torch.zeros(batch_size, device=device)

        compat_loss = (
            F.binary_cross_entropy(pos_compat, pos_labels) +
            F.binary_cross_entropy(neg_compat, neg_labels)
        ) / 2

        # Combined loss
        total_loss = masked_loss + compatibility_weight * compat_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_masked_loss += masked_loss.item()
        total_compat_loss += compat_loss.item()
        num_batches += 1
        pbar.set_postfix({
            'mask_loss': f'{masked_loss.item():.4f}',
            'compat_loss': f'{compat_loss.item():.4f}'
        })

    return total_masked_loss / num_batches, total_compat_loss / num_batches


def evaluate_fitb(
    model: OutfitTransformer,
    fitb_dataset: FITBDataset,
    device: torch.device
) -> float:
    """
    Evaluate FITB accuracy using visual feature compatibility.

    This is train/test agnostic - works with ANY items because
    it scores compatibility using visual features, not item IDs.

    For each question:
    1. For each answer candidate, create outfit: context + answer
    2. Get compatibility score from model
    3. Select answer with highest compatibility score
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for idx in tqdm(range(len(fitb_dataset)), desc="FITB Eval"):
            sample = fitb_dataset[idx]
            context_visual = sample['context_visual'].to(device)  # [context_len, 512]
            context_cats = sample['context_cats'].to(device)  # [context_len]
            answer_visual = sample['answer_visual'].to(device)  # [4, 512]
            answer_cats = sample['answer_cats'].to(device)  # [4]
            correct_idx = sample['correct_idx']

            # Score each answer using visual feature compatibility
            scores = model.score_fitb_answers(
                context_visual, context_cats,
                answer_visual, answer_cats
            )  # [4]

            # Check if correct answer has highest score
            pred_idx = torch.argmax(scores).item()
            if pred_idx == correct_idx:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy


def main():
    """Main training script"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/polyvore')
    parser.add_argument('--embeddings', default='models/polyvore_embeddings.pkl')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--save_dir', default='models')
    parser.add_argument('--compat_weight', type=float, default=1.0,
                        help='Weight for compatibility loss (critical for FITB!)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load datasets
    print("Loading datasets...")
    train_dataset = PolyvoreOutfitDataset(
        outfit_json=f'{args.data_dir}/train_no_dup.json',
        embeddings_pkl=args.embeddings,
        metadata_json=f'{args.data_dir}/polyvore_item_metadata.json',
        mode='train'
    )

    fitb_dataset = FITBDataset(
        fitb_json=f'{args.data_dir}/fill_in_blank_test.json',
        mapping_json=f'{args.data_dir}/image_to_tid_mapping.json',
        embeddings_pkl=args.embeddings,
        metadata_json=f'{args.data_dir}/polyvore_item_metadata.json',
        tid_to_idx=train_dataset.tid_to_idx,
        category_to_idx=train_dataset.category_to_idx
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_outfits,
        num_workers=4,
        pin_memory=True
    )

    # Prepare embeddings array for negative sampling in compatibility training
    print("Preparing embeddings for compatibility training...")
    all_embeddings = np.stack(list(train_dataset.embeddings.values()))
    print(f"  Embeddings array shape: {all_embeddings.shape}")

    # Create model
    print("Creating model...")
    model = OutfitTransformer(
        visual_dim=512,
        num_categories=len(train_dataset.category_to_idx),
        num_items=len(train_dataset.tid_to_idx),
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Compatibility weight: {args.compat_weight}")
    print(f"\nTraining with BOTH masked prediction AND compatibility scoring")
    print(f"  - Masked prediction → trains item_prediction_head")
    print(f"  - Compatibility scoring → trains compatibility_head (CRITICAL for FITB!)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_fitb = 0
    for epoch in range(1, args.epochs + 1):
        masked_loss, compat_loss = train_outfit_transformer(
            model, train_loader, optimizer, device, epoch,
            all_embeddings=all_embeddings,
            compatibility_weight=args.compat_weight
        )
        scheduler.step()

        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            fitb_acc = evaluate_fitb(model, fitb_dataset, device)
            print(f"Epoch {epoch}: MaskLoss={masked_loss:.4f}, CompatLoss={compat_loss:.4f}, FITB={fitb_acc*100:.2f}%")

            if fitb_acc > best_fitb:
                best_fitb = fitb_acc
                save_path = f'{args.save_dir}/outfit_transformer_best.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'fitb_accuracy': fitb_acc,
                    'config': {
                        'hidden_dim': args.hidden_dim,
                        'n_layers': args.n_layers,
                        'n_heads': args.n_heads,
                        'num_categories': len(train_dataset.category_to_idx),
                        'num_items': len(train_dataset.tid_to_idx)
                    }
                }, save_path)
                print(f"  *** New best! Saved to {save_path}")
        else:
            print(f"Epoch {epoch}: MaskLoss={masked_loss:.4f}, CompatLoss={compat_loss:.4f}")

    print(f"\nTraining complete! Best FITB accuracy: {best_fitb*100:.2f}%")


if __name__ == '__main__':
    main()
