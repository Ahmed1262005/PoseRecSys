"""
Evaluation Module for Fashion Personalized Feed
Implements metrics for outfit compatibility, FITB, and retrieval quality
"""
import json
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def load_test_outfits(test_path: str) -> Dict:
    """Load test outfit data"""
    with open(test_path, 'r') as f:
        return json.load(f)


def evaluate_compatibility_auc(
    test_path: str,
    embeddings_path: str,
    n_negative: int = 1
) -> float:
    """
    Evaluate outfit compatibility prediction using AUC

    For each outfit, we compute average pairwise similarity between items.
    Negative outfits are created by replacing items with random ones.

    Args:
        test_path: Path to test.json
        embeddings_path: Path to embeddings pickle
        n_negative: Number of negative samples per positive

    Returns:
        AUC score
    """
    from embeddings import FashionEmbeddingGenerator

    # Load data
    test_data = load_test_outfits(test_path)
    embeddings, item_ids = FashionEmbeddingGenerator.load_embeddings(embeddings_path)
    item_set = set(embeddings.keys())

    y_true = []
    y_scores = []

    print("Evaluating compatibility prediction...")
    for outfit_id, outfit_data in tqdm(test_data.items()):
        items = outfit_data.get('items', [])

        # Get item IDs
        outfit_item_ids = []
        for item in items:
            if isinstance(item, dict):
                item_id = item.get('item_id', item.get('index'))
            else:
                item_id = str(item)
            if item_id in embeddings:
                outfit_item_ids.append(item_id)

        if len(outfit_item_ids) < 2:
            continue

        # Positive: compute outfit compatibility score
        pos_score = compute_outfit_compatibility(outfit_item_ids, embeddings)
        y_true.append(1)
        y_scores.append(pos_score)

        # Negative: replace random items
        for _ in range(n_negative):
            neg_items = outfit_item_ids.copy()
            # Replace half the items
            n_replace = max(1, len(neg_items) // 2)
            indices_to_replace = random.sample(range(len(neg_items)), n_replace)

            for idx in indices_to_replace:
                # Sample random item not in outfit
                random_item = random.choice(list(item_set - set(neg_items)))
                neg_items[idx] = random_item

            neg_score = compute_outfit_compatibility(neg_items, embeddings)
            y_true.append(0)
            y_scores.append(neg_score)

    if not y_true:
        print("Warning: No valid outfits found for evaluation")
        return 0.0

    auc = roc_auc_score(y_true, y_scores)
    print(f"Compatibility AUC: {auc:.4f}")
    return auc


def compute_outfit_compatibility(
    item_ids: List[str],
    embeddings: Dict[str, np.ndarray]
) -> float:
    """
    Compute compatibility score for an outfit

    Uses average pairwise cosine similarity between items.

    Args:
        item_ids: List of item IDs in outfit
        embeddings: Dictionary of item embeddings

    Returns:
        Compatibility score (0-1)
    """
    if len(item_ids) < 2:
        return 0.5

    similarities = []
    for i in range(len(item_ids)):
        for j in range(i + 1, len(item_ids)):
            emb_i = embeddings.get(item_ids[i])
            emb_j = embeddings.get(item_ids[j])

            if emb_i is not None and emb_j is not None:
                sim = np.dot(emb_i, emb_j)
                similarities.append(sim)

    if not similarities:
        return 0.5

    return float(np.mean(similarities))


def evaluate_fitb_accuracy(
    test_path: str,
    embeddings_path: str,
    k: int = 10
) -> float:
    """
    Evaluate Fill-in-the-Blank (FITB) task accuracy

    Given an outfit with one item removed, try to predict the missing item.
    Accuracy is whether the correct item is in the top-k predictions.

    Args:
        test_path: Path to test.json
        embeddings_path: Path to embeddings pickle
        k: Top-k for accuracy calculation

    Returns:
        Accuracy score (0-1)
    """
    from embeddings import FaissSearcher

    # Load data
    test_data = load_test_outfits(test_path)
    searcher = FaissSearcher(embeddings_path, use_gpu=False)

    correct = 0
    total = 0

    print(f"Evaluating FITB accuracy (top-{k})...")
    for outfit_id, outfit_data in tqdm(test_data.items()):
        items = outfit_data.get('items', [])

        # Get item IDs
        outfit_item_ids = []
        for item in items:
            if isinstance(item, dict):
                item_id = item.get('item_id', item.get('index'))
            else:
                item_id = str(item)
            if item_id in searcher.embeddings:
                outfit_item_ids.append(item_id)

        if len(outfit_item_ids) < 3:
            continue

        # Remove one item and try to predict it
        for remove_idx in range(len(outfit_item_ids)):
            missing_item = outfit_item_ids[remove_idx]
            remaining_items = outfit_item_ids[:remove_idx] + outfit_item_ids[remove_idx + 1:]

            # Predict missing item using average of remaining items
            predictions = predict_fitb_item(remaining_items, searcher, k=k)
            predicted_ids = [p['item_id'] for p in predictions]

            if missing_item in predicted_ids:
                correct += 1
            total += 1

    if total == 0:
        print("Warning: No valid FITB samples found")
        return 0.0

    accuracy = correct / total
    print(f"FITB Accuracy (top-{k}): {accuracy:.4f}")
    return accuracy


def predict_fitb_item(
    context_items: List[str],
    searcher,
    k: int = 10
) -> List[Dict]:
    """
    Predict missing item given context items

    Uses average embedding of context items to find similar items.

    Args:
        context_items: List of context item IDs
        searcher: FaissSearcher instance
        k: Number of predictions

    Returns:
        List of predictions with item_id and score
    """
    # Compute average embedding
    embeddings = [searcher.embeddings[iid] for iid in context_items if iid in searcher.embeddings]

    if not embeddings:
        return []

    avg_embedding = np.mean(embeddings, axis=0)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

    # Search for similar items
    results = searcher.search_by_embedding(avg_embedding, k=k + len(context_items))

    # Exclude context items
    context_set = set(context_items)
    predictions = [r for r in results if r['item_id'] not in context_set][:k]

    return predictions


def evaluate_retrieval_metrics(
    test_path: str,
    embeddings_path: str,
    model_checkpoint: Optional[str] = None,
    topk: List[int] = [10, 20, 50]
) -> Dict[str, float]:
    """
    Evaluate retrieval quality metrics

    Uses RecBole evaluation if model checkpoint provided,
    otherwise uses visual similarity for evaluation.

    Args:
        test_path: Path to test.json
        embeddings_path: Path to embeddings
        model_checkpoint: Optional RecBole model checkpoint
        topk: List of k values for metrics

    Returns:
        Dictionary of metrics (recall@k, ndcg@k, hit@k)
    """
    if model_checkpoint:
        return evaluate_recbole_retrieval(model_checkpoint, topk)
    else:
        return evaluate_visual_retrieval(test_path, embeddings_path, topk)


def evaluate_visual_retrieval(
    test_path: str,
    embeddings_path: str,
    topk: List[int] = [10, 20, 50]
) -> Dict[str, float]:
    """
    Evaluate retrieval using visual similarity

    For each outfit, check if other items from the same outfit
    appear in the top-k similar items.

    Args:
        test_path: Path to test.json
        embeddings_path: Path to embeddings
        topk: List of k values

    Returns:
        Dictionary of metrics
    """
    from embeddings import FaissSearcher

    test_data = load_test_outfits(test_path)
    searcher = FaissSearcher(embeddings_path, use_gpu=False)

    max_k = max(topk)
    metrics = defaultdict(list)

    print("Evaluating visual retrieval metrics...")
    for outfit_id, outfit_data in tqdm(test_data.items()):
        items = outfit_data.get('items', [])

        outfit_item_ids = []
        for item in items:
            if isinstance(item, dict):
                item_id = item.get('item_id', item.get('index'))
            else:
                item_id = str(item)
            if item_id in searcher.embeddings:
                outfit_item_ids.append(item_id)

        if len(outfit_item_ids) < 2:
            continue

        # For each item, check if other outfit items are in top-k
        for query_idx, query_item in enumerate(outfit_item_ids):
            target_items = set(outfit_item_ids) - {query_item}

            results = searcher.search(query_item, k=max_k)
            retrieved_ids = [r['item_id'] for r in results]

            for k in topk:
                retrieved_at_k = set(retrieved_ids[:k])
                hits = len(target_items & retrieved_at_k)

                # Recall@k
                recall = hits / len(target_items) if target_items else 0
                metrics[f'recall@{k}'].append(recall)

                # Hit@k (binary)
                hit = 1 if hits > 0 else 0
                metrics[f'hit@{k}'].append(hit)

                # NDCG@k
                ndcg = compute_ndcg(retrieved_ids[:k], target_items)
                metrics[f'ndcg@{k}'].append(ndcg)

    # Average metrics
    results = {}
    for metric_name, values in metrics.items():
        results[metric_name] = np.mean(values) if values else 0.0
        print(f"{metric_name}: {results[metric_name]:.4f}")

    return results


def evaluate_recbole_retrieval(
    checkpoint_path: str,
    topk: List[int] = [10, 20, 50]
) -> Dict[str, float]:
    """
    Evaluate retrieval using RecBole model

    Uses RecBole's built-in evaluation on test set.

    Args:
        checkpoint_path: Path to model checkpoint
        topk: List of k values

    Returns:
        Dictionary of metrics
    """
    from train_models import evaluate_model

    metrics = evaluate_model(checkpoint_path)
    return metrics


def compute_ndcg(
    ranked_list: List[str],
    relevant_items: set,
    k: Optional[int] = None
) -> float:
    """
    Compute NDCG (Normalized Discounted Cumulative Gain)

    Args:
        ranked_list: List of item IDs in rank order
        relevant_items: Set of relevant item IDs
        k: Optional cutoff

    Returns:
        NDCG score (0-1)
    """
    if k is not None:
        ranked_list = ranked_list[:k]

    if not ranked_list or not relevant_items:
        return 0.0

    # DCG
    dcg = 0.0
    for i, item_id in enumerate(ranked_list):
        if item_id in relevant_items:
            dcg += 1.0 / np.log2(i + 2)  # +2 because i is 0-indexed

    # Ideal DCG
    ideal_length = min(len(ranked_list), len(relevant_items))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_length))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_fitb_polyvore_u(
    fitb_test_path: str,
    candidates_path: str,
    embeddings_path: Optional[str] = None,
    bert4rec_checkpoint: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Evaluate FITB using Polyvore-U test format

    Polyvore-U FITB format:
    - fitb_test.npy: outfits with one item masked (id=0)
    - fitb_test_retrieval_candidates.npy: 5 candidates per outfit (first is ground truth)

    Scoring modes:
    - BERT4Rec only: Use sequential prediction scores
    - Visual only: Use FashionCLIP similarity (requires matching embeddings)
    - Hybrid: Combine both

    Args:
        fitb_test_path: Path to fitb_test.npy
        candidates_path: Path to fitb_test_retrieval_candidates.npy
        embeddings_path: Optional path to embeddings (for visual scoring)
        bert4rec_checkpoint: Optional BERT4Rec checkpoint for sequential scoring
        weights: Scoring weights {'visual': 0.5, 'sequential': 0.5}

    Returns:
        Dict with accuracy metrics
    """
    import torch

    if weights is None:
        weights = {'visual': 0.0, 'sequential': 1.0}  # Default to BERT4Rec only

    # Load Polyvore-U FITB test data
    fitb_data = np.load(fitb_test_path, allow_pickle=True).item()
    candidates_data = np.load(candidates_path, allow_pickle=True).item()

    # Load embeddings if provided and visual weight > 0
    embeddings = {}
    if embeddings_path and weights.get('visual', 0) > 0:
        try:
            from embeddings import FashionEmbeddingGenerator
            embeddings, _ = FashionEmbeddingGenerator.load_embeddings(embeddings_path)
            print(f"Loaded {len(embeddings)} embeddings for visual scoring")
        except Exception as e:
            print(f"Warning: Could not load embeddings: {e}")

    # Load BERT4Rec model if provided
    bert4rec_model = None
    bert4rec_config = None
    bert4rec_dataset = None
    bert4rec_token2id = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if bert4rec_checkpoint and weights.get('sequential', 0) > 0:
        try:
            # Use custom loading to handle PyTorch 2.6 weights_only issue
            import torch
            checkpoint = torch.load(bert4rec_checkpoint, map_location=device, weights_only=False)

            # Reconstruct config and model from checkpoint
            config = checkpoint['config']
            from recbole.model.sequential_recommender import BERT4Rec
            from recbole.data import create_dataset, data_preparation

            # Create dataset to get vocabulary mapping
            dataset = create_dataset(config)
            bert4rec_dataset = dataset
            bert4rec_config = config

            # Create model and load state dict
            bert4rec_model = BERT4Rec(config, dataset).to(device)
            bert4rec_model.load_state_dict(checkpoint['state_dict'])
            bert4rec_model.eval()

            bert4rec_token2id = dataset.field2token_id['item_id']
            print(f"Loaded BERT4Rec for FITB evaluation (vocab size: {len(bert4rec_token2id)})")
        except Exception as e:
            print(f"Warning: Could not load BERT4Rec: {e}")
            import traceback
            traceback.print_exc()

    correct = 0
    total = 0
    skipped_no_context = 0
    skipped_no_cands = 0

    uids = fitb_data['uids']
    oids = fitb_data['oids']
    outfits = fitb_data['outfits']
    categories = fitb_data.get('category', [None] * len(uids))

    print(f"Evaluating FITB on {len(uids)} samples...")
    print(f"  Weights: visual={weights.get('visual', 0)}, sequential={weights.get('sequential', 0)}")

    for i, (uid, oid, outfit, cats) in enumerate(tqdm(zip(uids, oids, outfits, categories), total=len(uids))):
        # Get candidates for this outfit
        if uid not in candidates_data or oid not in candidates_data[uid]:
            skipped_no_cands += 1
            continue

        candidates = candidates_data[uid][oid]  # [ground_truth, neg1, neg2, neg3, neg4]
        if len(candidates) < 2:
            skipped_no_cands += 1
            continue

        ground_truth = str(candidates[0])

        # Find masked position (item_id = 0) and context items
        context_items = []
        masked_idx = -1
        for idx, item_id in enumerate(outfit):
            if item_id == 0:
                masked_idx = idx
            else:
                context_items.append(str(item_id))

        if masked_idx == -1 or not context_items:
            skipped_no_context += 1
            continue

        # Score each candidate
        candidate_scores = []
        for cand in candidates:
            cand_str = str(cand)
            score = 0.0
            has_score = False

            # Visual score: similarity to context items
            if weights.get('visual', 0) > 0 and embeddings:
                if cand_str in embeddings:
                    visual_scores = []
                    for ctx_item in context_items:
                        if ctx_item in embeddings:
                            sim = np.dot(embeddings[cand_str], embeddings[ctx_item])
                            visual_scores.append(sim)
                    if visual_scores:
                        score += weights['visual'] * np.mean(visual_scores)
                        has_score = True

            # Sequential score: BERT4Rec prediction
            if weights.get('sequential', 0) > 0 and bert4rec_model is not None:
                seq_score = get_bert4rec_score(
                    context_items, cand_str,
                    bert4rec_model, bert4rec_config, bert4rec_dataset,
                    bert4rec_token2id, device
                )
                if seq_score is not None:
                    score += weights['sequential'] * seq_score
                    has_score = True

            candidate_scores.append((cand_str, score, has_score))

        # Check if we have valid scores
        valid_scores = [(c, s) for c, s, h in candidate_scores if h]
        if not valid_scores:
            # Random baseline if no scores available
            import random
            predicted = random.choice([str(c) for c in candidates])
        else:
            # Rank candidates by score
            valid_scores.sort(key=lambda x: x[1], reverse=True)
            predicted = valid_scores[0][0]

        if predicted == ground_truth:
            correct += 1

        total += 1

    accuracy = correct / total if total > 0 else 0.0
    random_baseline = 1.0 / 5  # 5 candidates

    print(f"\nFITB Results:")
    print(f"  Total samples: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Random baseline: {random_baseline:.4f} ({random_baseline*100:.2f}%)")
    print(f"  Lift over random: {accuracy/random_baseline:.2f}x")
    if skipped_no_context > 0:
        print(f"  Skipped (no context): {skipped_no_context}")
    if skipped_no_cands > 0:
        print(f"  Skipped (no candidates): {skipped_no_cands}")

    return {
        'fitb_accuracy': accuracy,
        'fitb_total': total,
        'fitb_correct': correct,
        'random_baseline': random_baseline
    }


def get_bert4rec_score(
    context_items: List[str],
    target_item: str,
    model,
    config,
    dataset,
    token2id: Dict,
    device
) -> Optional[float]:
    """
    Get BERT4Rec score for target item given context

    Args:
        context_items: List of context item IDs
        target_item: Item to score
        model: BERT4Rec model
        config: Model config
        dataset: RecBole dataset
        token2id: Item token to ID mapping
        device: Torch device

    Returns:
        Score or None if items not in vocabulary
    """
    import torch

    try:
        max_len = config['MAX_ITEM_LIST_LENGTH']

        # Convert context items to internal IDs
        valid_items = []
        for item_id in context_items:
            if str(item_id) in token2id:
                valid_items.append(token2id[str(item_id)])

        if not valid_items:
            return None

        # Check if target is in vocabulary
        if str(target_item) not in token2id:
            return None
        target_id = token2id[str(target_item)]

        # Truncate to max length
        if len(valid_items) > max_len - 1:
            valid_items = valid_items[-(max_len - 1):]

        # Create sequence with mask at end
        item_seq = torch.zeros(max_len, dtype=torch.long, device=device)
        item_seq_len = len(valid_items) + 1

        start_idx = max_len - item_seq_len
        for i, item_id in enumerate(valid_items):
            item_seq[start_idx + i] = item_id

        # Mask token
        mask_token = dataset.item_num
        item_seq[max_len - 1] = mask_token

        # Forward pass
        item_seq = item_seq.unsqueeze(0)
        item_seq_len = torch.tensor([item_seq_len], device=device)

        with torch.no_grad():
            output = model.forward(item_seq, item_seq_len)
            # Get score for target item at masked position
            score = output[0, -1, target_id].item()

        return score

    except Exception as e:
        return None


def run_full_evaluation(
    data_dir: str = "data/polyvore",
    embeddings_path: str = "models/polyvore_embeddings.pkl",
    model_checkpoint: Optional[str] = None,
    output_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Run complete evaluation suite

    Args:
        data_dir: Path to data directory
        embeddings_path: Path to embeddings
        model_checkpoint: Optional RecBole checkpoint
        output_path: Optional path to save results

    Returns:
        Dictionary of all metrics
    """
    import os

    test_path = os.path.join(data_dir, "test.json")
    if not os.path.exists(test_path):
        test_path = os.path.join(data_dir, "polyvore_outfits", "test.json")

    results = {}

    print("\n" + "=" * 50)
    print("RUNNING FULL EVALUATION")
    print("=" * 50)

    # 1. Compatibility AUC
    print("\n1. Compatibility Prediction (AUC)")
    print("-" * 30)
    try:
        auc = evaluate_compatibility_auc(test_path, embeddings_path)
        results['compatibility_auc'] = auc
    except Exception as e:
        print(f"Error: {e}")
        results['compatibility_auc'] = None

    # 2. FITB Accuracy
    print("\n2. Fill-in-the-Blank Accuracy")
    print("-" * 30)
    try:
        fitb_acc = evaluate_fitb_accuracy(test_path, embeddings_path, k=10)
        results['fitb_accuracy'] = fitb_acc
    except Exception as e:
        print(f"Error: {e}")
        results['fitb_accuracy'] = None

    # 3. Retrieval Metrics
    print("\n3. Retrieval Metrics")
    print("-" * 30)
    try:
        retrieval = evaluate_retrieval_metrics(test_path, embeddings_path, model_checkpoint)
        results.update(retrieval)
    except Exception as e:
        print(f"Error: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    for metric, value in results.items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: N/A")

    # Check targets
    print("\n" + "-" * 30)
    print("Target Achievement:")
    if results.get('compatibility_auc'):
        status = "PASS" if results['compatibility_auc'] > 0.85 else "FAIL"
        print(f"  AUC > 0.85: {status}")
    if results.get('fitb_accuracy'):
        status = "PASS" if results['fitb_accuracy'] > 0.60 else "FAIL"
        print(f"  FITB > 60%: {status}")
    if results.get('ndcg@10'):
        status = "PASS" if results['ndcg@10'] > 0.40 else "FAIL"
        print(f"  NDCG@10 > 0.40: {status}")

    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def run_polyvore_u_evaluation(
    polyvore_u_dir: str = "data/polyvore_u",
    embeddings_path: Optional[str] = None,
    bert4rec_checkpoint: Optional[str] = None,
    output_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Run evaluation on Polyvore-U dataset

    Args:
        polyvore_u_dir: Path to Polyvore-U data directory
        embeddings_path: Path to FashionCLIP embeddings (optional)
        bert4rec_checkpoint: Path to BERT4Rec checkpoint
        output_path: Optional path to save results

    Returns:
        Dictionary of all metrics
    """
    import os

    results = {}

    print("\n" + "=" * 60)
    print("POLYVORE-U EVALUATION")
    print("=" * 60)

    # FITB Test paths
    fitb_test_path = os.path.join(polyvore_u_dir, "fitb_test.npy")
    candidates_path = os.path.join(polyvore_u_dir, "fitb_test_retrieval_candidates.npy")

    if not os.path.exists(fitb_test_path):
        print(f"Warning: {fitb_test_path} not found")
        return results

    if not os.path.exists(candidates_path):
        print(f"Warning: {candidates_path} not found")
        return results

    # Run BERT4Rec-only FITB evaluation
    if bert4rec_checkpoint and os.path.exists(bert4rec_checkpoint):
        print("\n1. FITB with BERT4Rec Sequential Model")
        print("-" * 40)
        seq_results = evaluate_fitb_polyvore_u(
            fitb_test_path, candidates_path,
            embeddings_path=None,
            bert4rec_checkpoint=bert4rec_checkpoint,
            weights={'visual': 0.0, 'sequential': 1.0}
        )
        results['fitb_bert4rec'] = seq_results['fitb_accuracy']
        results['fitb_total'] = seq_results['fitb_total']
        results['fitb_correct'] = seq_results['fitb_correct']
    else:
        print("No BERT4Rec checkpoint provided - skipping sequential evaluation")

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f} ({value*100:.2f}%)")
        else:
            print(f"  {metric}: {value}")

    # Target check
    print("\n" + "-" * 40)
    print("Target Achievement:")
    fitb_acc = results.get('fitb_bert4rec', 0)
    random_baseline = 0.20  # 1/5 candidates
    status = "PASS" if fitb_acc > 0.60 else "FAIL"
    print(f"  FITB > 60%: {status} (actual: {fitb_acc*100:.2f}%)")
    print(f"  Random baseline: 20.00%")
    print(f"  Lift over random: {fitb_acc/random_baseline:.2f}x")

    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    import sys

    # Determine evaluation mode
    mode = sys.argv[1] if len(sys.argv) > 1 else "polyvore_u"

    if mode == "polyvore_u":
        # Polyvore-U evaluation with BERT4Rec
        polyvore_u_dir = sys.argv[2] if len(sys.argv) > 2 else "data/polyvore_u"
        embeddings_path = sys.argv[3] if len(sys.argv) > 3 else "models/polyvore_embeddings.pkl"
        bert4rec_checkpoint = sys.argv[4] if len(sys.argv) > 4 else None

        # Auto-find checkpoint if not provided
        if not bert4rec_checkpoint:
            from pathlib import Path
            checkpoints = list(Path("models").glob("BERT4Rec*.pth"))
            if checkpoints:
                bert4rec_checkpoint = str(max(checkpoints, key=lambda p: p.stat().st_mtime))
                print(f"Auto-found BERT4Rec checkpoint: {bert4rec_checkpoint}")

        run_polyvore_u_evaluation(
            polyvore_u_dir=polyvore_u_dir,
            embeddings_path=embeddings_path,
            bert4rec_checkpoint=bert4rec_checkpoint,
            output_path="evaluation_results_polyvore_u.json"
        )
    else:
        # Original Polyvore evaluation
        data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/polyvore"
        embeddings_path = sys.argv[2] if len(sys.argv) > 2 else "models/polyvore_embeddings.pkl"
        checkpoint = sys.argv[3] if len(sys.argv) > 3 else None

        run_full_evaluation(
            data_dir=data_dir,
            embeddings_path=embeddings_path,
            model_checkpoint=checkpoint,
            output_path="evaluation_results.json"
        )
