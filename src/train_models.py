"""
RecBole Model Training Module
Train collaborative filtering models on Polyvore data
"""
import os
from typing import Dict, Any, Optional
from pathlib import Path


def get_bpr_config(data_path: str = "data", dataset: str = "polyvore") -> Dict[str, Any]:
    """
    Get BPR (Bayesian Personalized Ranking) model configuration

    Args:
        data_path: Path to data directory
        dataset: Dataset name

    Returns:
        Configuration dictionary for RecBole
    """
    return {
        # Dataset
        'data_path': data_path,
        'dataset': dataset,
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'RATING_FIELD': 'rating',
        'TIME_FIELD': 'timestamp',
        'load_col': {
            'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
        },

        # Split configuration
        'eval_args': {
            'split': {'RS': [0.8, 0.1, 0.1]},
            'group_by': 'user',
            'order': 'TO',  # Time-ordered
            'mode': 'full'
        },

        # Model hyperparameters
        'embedding_size': 64,
        'learning_rate': 0.001,

        # Training settings
        'epochs': 100,
        'train_batch_size': 4096,
        'eval_batch_size': 4096,
        'stopping_step': 10,
        'neg_sampling': {
            'uniform': 1
        },

        # Evaluation
        'metrics': ['Recall', 'NDCG', 'Hit', 'Precision'],
        'topk': [10, 20, 50],
        'valid_metric': 'NDCG@10',

        # System
        'gpu_id': '0',
        'seed': 2024,
        'checkpoint_dir': 'models',
        'show_progress': True,

        # Logging
        'log_wandb': False,
        'state': 'INFO',
    }


def get_lightgcn_config(data_path: str = "data", dataset: str = "polyvore") -> Dict[str, Any]:
    """
    Get LightGCN model configuration (graph-based collaborative filtering)

    Args:
        data_path: Path to data directory
        dataset: Dataset name

    Returns:
        Configuration dictionary for RecBole
    """
    config = get_bpr_config(data_path, dataset)

    # LightGCN specific parameters
    config.update({
        'n_layers': 3,
        'reg_weight': 1e-4,
    })

    return config


def get_duorec_config(data_path: str = "data", dataset: str = "amazon_mens") -> Dict[str, Any]:
    """
    Get DuoRec sequential model configuration (Contrastive Learning)

    DuoRec (WSDM 2022) uses contrastive learning to address representation
    degeneration in sequential recommendation.

    Paper: "Contrastive Learning for Representation Degeneration Problem
           in Sequential Recommendation"

    Args:
        data_path: Path to data directory
        dataset: Dataset name

    Returns:
        Configuration dictionary for RecBole
    """
    return {
        # Dataset
        'data_path': data_path,
        'dataset': dataset,
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'load_col': {
            'inter': ['user_id', 'item_id', 'timestamp'],
        },

        # Sequential model settings
        'MAX_ITEM_LIST_LENGTH': 50,
        'train_neg_sample_args': None,

        # DuoRec architecture (based on Transformer)
        'n_layers': 2,
        'n_heads': 2,
        'hidden_size': 64,
        'inner_size': 256,
        'hidden_dropout_prob': 0.5,
        'attn_dropout_prob': 0.5,
        'hidden_act': 'gelu',
        'layer_norm_eps': 1e-12,

        # DuoRec contrastive learning params
        'lmd': 0.1,              # Weight for contrastive loss
        'tau': 1.0,              # Temperature for softmax
        'sim': 'dot',            # Similarity function (dot or cos)
        'model_augmentation': True,  # Use model-level augmentation

        # Split configuration
        'eval_args': {
            'split': {'RS': [0.8, 0.1, 0.1]},
            'group_by': 'user',
            'order': 'TO',
            'mode': 'full'
        },

        # Training settings
        'learning_rate': 0.001,
        'epochs': 200,
        'train_batch_size': 256,
        'eval_batch_size': 2048,
        'stopping_step': 20,

        # Evaluation
        'metrics': ['Recall', 'NDCG', 'Hit', 'MRR'],
        'topk': [5, 10, 20],
        'valid_metric': 'NDCG@10',

        # System
        'gpu_id': '0',
        'seed': 2024,
        'checkpoint_dir': 'models',
        'show_progress': True,

        # Logging
        'log_wandb': False,
        'state': 'INFO',
    }


def get_bert4rec_config(data_path: str = "data", dataset: str = "polyvore_u") -> Dict[str, Any]:
    """
    Get BERT4Rec sequential model configuration (LARGE model for 25GB GPU)

    BERT4Rec uses bidirectional self-attention for sequential recommendation.
    It masks items in the sequence and predicts them using context from both sides.

    Args:
        data_path: Path to data directory
        dataset: Dataset name

    Returns:
        Configuration dictionary for RecBole
    """
    return {
        # Dataset
        'data_path': data_path,
        'dataset': dataset,
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'load_col': {
            'inter': ['user_id', 'item_id', 'timestamp'],
        },

        # Sequential model settings
        'MAX_ITEM_LIST_LENGTH': 50,  # Max items per user sequence
        'train_neg_sample_args': None,  # BERT4Rec uses masked LM, not negative sampling

        # BERT4Rec architecture - SCALED UP for 25GB GPU
        'n_layers': 4,           # Number of Transformer layers (was 2)
        'n_heads': 8,            # Number of attention heads (was 2)
        'hidden_size': 256,      # Hidden dimension (was 64)
        'inner_size': 1024,      # Feed-forward inner dimension (was 256)
        'hidden_dropout_prob': 0.3,  # Dropout for hidden layers (reduced for larger model)
        'attn_dropout_prob': 0.3,    # Dropout for attention
        'hidden_act': 'gelu',    # Activation function
        'layer_norm_eps': 1e-12, # Layer norm epsilon
        'mask_ratio': 0.2,       # Ratio of items to mask during training
        'loss_type': 'CE',       # Cross-entropy loss

        # Split configuration
        'eval_args': {
            'split': {'RS': [0.8, 0.1, 0.1]},
            'group_by': 'user',
            'order': 'TO',  # Time-ordered for sequential
            'mode': 'full'
        },

        # Training settings - SCALED UP
        'learning_rate': 0.0005,  # Slightly lower for larger model
        'epochs': 300,            # More epochs
        'train_batch_size': 1024,  # Larger batch (was 256)
        'eval_batch_size': 2048,   # Larger eval batch
        'stopping_step': 30,       # More patience

        # Evaluation
        'metrics': ['Recall', 'NDCG', 'Hit', 'MRR'],
        'topk': [1, 5, 10, 20],
        'valid_metric': 'NDCG@10',

        # System
        'gpu_id': '0',
        'seed': 2024,
        'checkpoint_dir': 'models',
        'show_progress': True,

        # Logging
        'log_wandb': False,
        'state': 'INFO',
    }


def train_bert4rec(
    data_path: str = "data/polyvore_u_recbole",
    dataset: str = "polyvore_u",
    save_model: bool = True
) -> Dict[str, Any]:
    """
    Train BERT4Rec sequential recommendation model

    Args:
        data_path: Path to RecBole format data
        dataset: Dataset name
        save_model: Whether to save checkpoint

    Returns:
        Dictionary with training results
    """
    config = get_bert4rec_config(data_path, dataset)

    print("Training BERT4Rec sequential model...")
    print(f"  Dataset: {dataset}")
    print(f"  Data path: {data_path}")
    print(f"  Max sequence length: {config['MAX_ITEM_LIST_LENGTH']}")
    print(f"  Mask ratio: {config['mask_ratio']}")
    print(f"  Architecture: {config['n_layers']} layers, {config['n_heads']} heads, hidden={config['hidden_size']}")

    return train_model('BERT4Rec', config, save_model)


def train_model(
    model_name: str,
    config_dict: Optional[Dict[str, Any]] = None,
    save_model: bool = True
) -> Dict[str, Any]:
    """
    Train a RecBole model

    Args:
        model_name: Name of the model (e.g., 'BPR', 'LightGCN')
        config_dict: Configuration dictionary (uses default if None)
        save_model: Whether to save the trained model

    Returns:
        Dictionary with training results and metrics
    """
    from recbole.quick_start import run_recbole

    if config_dict is None:
        if model_name == 'LightGCN':
            config_dict = get_lightgcn_config()
        else:
            config_dict = get_bpr_config()

    print(f"Training {model_name} model...")
    print(f"  Dataset: {config_dict.get('dataset')}")
    print(f"  Data path: {config_dict.get('data_path')}")
    print(f"  Epochs: {config_dict.get('epochs')}")

    # Run training
    result = run_recbole(
        model=model_name,
        config_dict=config_dict
    )

    # Parse results
    output = {
        'model_name': model_name,
        'best_valid_score': result.get('best_valid_score', 0),
        'valid_score_bigger': result.get('valid_score_bigger', True),
        'best_valid_result': result.get('best_valid_result', {}),
        'test_result': result.get('test_result', {}),
    }

    print(f"\nTraining complete!")
    print(f"  Best valid score: {output['best_valid_score']:.4f}")

    if 'test_result' in result and result['test_result']:
        print("  Test results:")
        for metric, value in result['test_result'].items():
            if isinstance(value, float):
                print(f"    {metric}: {value:.4f}")

    return output


def evaluate_model(checkpoint_path: str, config_dict: Optional[Dict] = None) -> Dict[str, float]:
    """
    Evaluate a trained model

    Args:
        checkpoint_path: Path to model checkpoint
        config_dict: Configuration dictionary

    Returns:
        Dictionary of evaluation metrics
    """
    from recbole.quick_start import load_data_and_model
    from recbole.utils import get_trainer

    # Load model
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        checkpoint_path
    )

    # Get trainer
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # Evaluate on test set
    test_result = trainer.evaluate(test_data)

    # Format results
    metrics = {}
    for key, value in test_result.items():
        # Normalize metric names
        metric_name = key.lower().replace('@', '@')
        if isinstance(value, float):
            metrics[metric_name] = value

    return metrics


def get_model_for_inference(checkpoint_path: str):
    """
    Load model for inference

    Args:
        checkpoint_path: Path to model checkpoint

    Returns:
        Tuple of (config, model, dataset)
    """
    from recbole.quick_start import load_data_and_model

    config, model, dataset, _, _, _ = load_data_and_model(checkpoint_path)

    model.eval()

    return config, model, dataset


def get_user_recommendations(
    checkpoint_path: str,
    user_id: str,
    k: int = 20,
    exclude_history: bool = True
) -> list:
    """
    Get top-k recommendations for a user

    Args:
        checkpoint_path: Path to model checkpoint
        user_id: User ID to get recommendations for
        k: Number of recommendations
        exclude_history: Whether to exclude items user has interacted with

    Returns:
        List of dicts with 'item_id' and 'score'
    """
    import torch
    from recbole.quick_start import load_data_and_model

    config, model, dataset, train_data, _, _ = load_data_and_model(checkpoint_path)
    model.eval()

    # Get internal user ID
    user_token = dataset.field2token_id['user_id'].get(user_id)
    if user_token is None:
        print(f"Warning: User {user_id} not found in dataset")
        return []

    # Get all item embeddings
    with torch.no_grad():
        # Create interaction for prediction
        item_ids = torch.arange(dataset.item_num).to(model.device)
        user_ids = torch.full((dataset.item_num,), user_token).to(model.device)

        # Get scores
        if hasattr(model, 'predict'):
            # Standard prediction interface
            interaction = {
                'user_id': user_ids,
                'item_id': item_ids,
            }
            scores = model.predict(interaction)
        elif hasattr(model, 'full_sort_predict'):
            # Full ranking interface
            scores = model.full_sort_predict({'user_id': torch.tensor([user_token]).to(model.device)})
            scores = scores[0]
        else:
            # Fallback: compute scores manually
            user_emb = model.user_embedding(torch.tensor([user_token]).to(model.device))
            item_emb = model.item_embedding.weight
            scores = torch.matmul(user_emb, item_emb.T)[0]

        scores = scores.cpu().numpy()

    # Get user history if excluding
    history_items = set()
    if exclude_history:
        uid_series = train_data.dataset.inter_feat['user_id']
        iid_series = train_data.dataset.inter_feat['item_id']
        for uid, iid in zip(uid_series.numpy(), iid_series.numpy()):
            if uid == user_token:
                history_items.add(iid)

    # Get top-k items
    id2token = dataset.field2id_token['item_id']
    results = []

    sorted_indices = scores.argsort()[::-1]
    for idx in sorted_indices:
        if idx == 0:  # Skip padding
            continue
        if exclude_history and idx in history_items:
            continue

        item_token = id2token[idx]
        results.append({
            'item_id': item_token,
            'score': float(scores[idx])
        })

        if len(results) >= k:
            break

    return results


def find_latest_checkpoint(model_name: str, checkpoint_dir: str = "models") -> Optional[str]:
    """
    Find the latest checkpoint for a model

    Args:
        model_name: Name of the model
        checkpoint_dir: Directory to search

    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None

    # Find matching checkpoints
    checkpoints = list(checkpoint_path.glob(f"{model_name}*.pth"))

    if not checkpoints:
        return None

    # Return most recent
    return str(max(checkpoints, key=lambda p: p.stat().st_mtime))


def train_duorec(
    data_path: str = "data/amazon_fashion/recbole",
    dataset: str = "amazon_mens",
    save_model: bool = True
) -> Dict[str, Any]:
    """
    Train DuoRec contrastive sequential recommendation model

    Args:
        data_path: Path to RecBole format data
        dataset: Dataset name
        save_model: Whether to save checkpoint

    Returns:
        Dictionary with training results
    """
    config = get_duorec_config(data_path, dataset)

    print("Training DuoRec contrastive sequential model...")
    print(f"  Dataset: {dataset}")
    print(f"  Data path: {data_path}")
    print(f"  Max sequence length: {config['MAX_ITEM_LIST_LENGTH']}")
    print(f"  Contrastive weight (lmd): {config['lmd']}")
    print(f"  Temperature (tau): {config['tau']}")
    print(f"  Architecture: {config['n_layers']} layers, {config['n_heads']} heads, hidden={config['hidden_size']}")

    return train_model('DuoRec', config, save_model)


def get_sasrec_config(data_path: str = "data", dataset: str = "amazon_mens") -> Dict[str, Any]:
    """
    Get SASRec (Self-Attentive Sequential Recommendation) model configuration

    LARGE CONFIG - Optimized for 22GB GPU utilization (~15-18GB target)

    SASRec uses unidirectional self-attention for sequential recommendation.
    Similar to BERT4Rec but predicts the next item instead of masked items.

    Args:
        data_path: Path to data directory
        dataset: Dataset name

    Returns:
        Configuration dictionary for RecBole
    """
    return {
        # Dataset
        'data_path': data_path,
        'dataset': dataset,
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'load_col': {
            'inter': ['user_id', 'item_id', 'timestamp'],
        },

        # Sequential model settings
        'MAX_ITEM_LIST_LENGTH': 50,  # Max items per user sequence
        'train_neg_sample_args': None,  # SASRec uses CE loss, not negative sampling

        # SASRec architecture - SCALED UP for 22GB GPU
        'n_layers': 4,           # Number of Transformer layers (was 2)
        'n_heads': 8,            # Number of attention heads (was 2)
        'hidden_size': 256,      # Hidden dimension (was 64)
        'inner_size': 1024,      # Feed-forward inner dimension (was 256)
        'hidden_dropout_prob': 0.3,  # Reduced dropout for larger model
        'attn_dropout_prob': 0.3,
        'hidden_act': 'gelu',
        'layer_norm_eps': 1e-12,
        'loss_type': 'CE',       # Cross-entropy loss

        # Split configuration
        'eval_args': {
            'split': {'RS': [0.8, 0.1, 0.1]},
            'group_by': 'user',
            'order': 'TO',  # Time-ordered for sequential
            'mode': 'full'
        },

        # Training settings - SCALED UP batch sizes
        'learning_rate': 0.001,   # Higher LR
        'epochs': 200,
        'train_batch_size': 2048,  # Much larger batch (was 256)
        'eval_batch_size': 4096,   # Larger eval batch (was 2048)
        'stopping_step': 20,
        'shuffle': False,          # Keep time order, don't randomize

        # Evaluation
        'metrics': ['Recall', 'NDCG', 'Hit', 'MRR'],
        'topk': [5, 10, 20],
        'valid_metric': 'NDCG@10',

        # System
        'gpu_id': '0',
        'seed': 2024,
        'checkpoint_dir': 'models',
        'show_progress': True,

        # Logging
        'log_wandb': False,
        'state': 'INFO',
    }


def train_sasrec(
    data_path: str = "data/amazon_fashion/recbole",
    dataset: str = "amazon_mens",
    save_model: bool = True
) -> Dict[str, Any]:
    """
    Train SASRec sequential recommendation model

    Args:
        data_path: Path to RecBole format data
        dataset: Dataset name
        save_model: Whether to save checkpoint

    Returns:
        Dictionary with training results
    """
    config = get_sasrec_config(data_path, dataset)

    print("Training SASRec sequential model...")
    print(f"  Dataset: {dataset}")
    print(f"  Data path: {data_path}")
    print(f"  Max sequence length: {config['MAX_ITEM_LIST_LENGTH']}")
    print(f"  Architecture: {config['n_layers']} layers, {config['n_heads']} heads, hidden={config['hidden_size']}")

    return train_model('SASRec', config, save_model)


if __name__ == "__main__":
    import sys

    model_name = sys.argv[1] if len(sys.argv) > 1 else "BPR"
    data_path = sys.argv[2] if len(sys.argv) > 2 else "data"
    dataset = sys.argv[3] if len(sys.argv) > 3 else "polyvore"

    print(f"Training {model_name} on {dataset}...")

    if model_name == "BERT4Rec":
        # Use BERT4Rec-specific training
        result = train_bert4rec(data_path, dataset)
    elif model_name == "SASRec":
        # Use SASRec-specific training
        result = train_sasrec(data_path, dataset)
    elif model_name == "DuoRec":
        # Use DuoRec-specific training
        result = train_duorec(data_path, dataset)
    elif model_name == "LightGCN":
        config = get_lightgcn_config(data_path, dataset)
        result = train_model(model_name, config)
    else:
        config = get_bpr_config(data_path, dataset)
        result = train_model(model_name, config)

    print("\nFinal Results:")
    for key, value in result.items():
        print(f"  {key}: {value}")
