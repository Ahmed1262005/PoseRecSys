"""
Train SASRec on Amazon Men's Fashion dataset.
Uses standard RecBole - faster and more stable than DuoRec.
"""

import os
import sys

# Use standard RecBole, not DuoRec fork
from recbole.quick_start import run_recbole
from recbole.config import Config


def get_sasrec_config():
    """SASRec config optimized for 22GB GPU."""
    return {
        # Dataset
        'data_path': '/home/ubuntu/recSys/outfitTransformer/data/amazon_fashion/recbole',
        'dataset': 'amazon_mens',
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},

        # Sequential model settings
        'MAX_ITEM_LIST_LENGTH': 50,
        'train_neg_sample_args': None,  # SASRec uses CE loss

        # Model architecture - scaled for GPU
        'n_layers': 4,
        'n_heads': 8,
        'hidden_size': 256,
        'inner_size': 1024,
        'hidden_dropout_prob': 0.3,
        'attn_dropout_prob': 0.3,
        'hidden_act': 'gelu',
        'layer_norm_eps': 1e-12,
        'loss_type': 'CE',

        # Training settings - maximized batch sizes
        'learning_rate': 0.001,
        'epochs': 100,
        'train_batch_size': 2048,
        'eval_batch_size': 4096,
        'stopping_step': 15,
        'eval_step': 1,

        # Evaluation settings
        'eval_setting': 'TO_LS,full',
        'metrics': ['Recall', 'MRR', 'NDCG', 'Hit'],
        'valid_metric': 'NDCG@10',
        'topk': [5, 10, 20],

        # System
        'gpu_id': 0,
        'use_gpu': True,
        'show_progress': True,
        'checkpoint_dir': '/home/ubuntu/recSys/outfitTransformer/models/sasrec_amazon/',
    }


if __name__ == "__main__":
    os.makedirs('/home/ubuntu/recSys/outfitTransformer/models/sasrec_amazon/', exist_ok=True)

    config_dict = get_sasrec_config()

    print("=" * 60)
    print("Training SASRec on Amazon Men's Fashion")
    print("=" * 60)
    print(f"Dataset: {config_dict['dataset']}")
    print(f"Batch size: {config_dict['train_batch_size']}")
    print(f"Epochs: {config_dict['epochs']}")
    print("=" * 60)

    run_recbole(
        model='SASRec',
        dataset=config_dict['dataset'],
        config_dict=config_dict
    )
