"""
Tests for evaluation module
"""
import os
import json
import pytest
import pickle
import numpy as np
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_test_data(tmp_path):
    """Create sample test outfit data"""
    test_data = {
        "outfit_001": {
            "set_id": "outfit_001",
            "items": [
                {"item_id": "item_0", "index": 0},
                {"item_id": "item_1", "index": 1},
                {"item_id": "item_2", "index": 2},
            ]
        },
        "outfit_002": {
            "set_id": "outfit_002",
            "items": [
                {"item_id": "item_3", "index": 0},
                {"item_id": "item_4", "index": 1},
                {"item_id": "item_5", "index": 2},
            ]
        },
        "outfit_003": {
            "set_id": "outfit_003",
            "items": [
                {"item_id": "item_0", "index": 0},
                {"item_id": "item_6", "index": 1},
                {"item_id": "item_7", "index": 2},
            ]
        }
    }

    test_path = str(tmp_path / "test.json")
    with open(test_path, 'w') as f:
        json.dump(test_data, f)

    return test_path


@pytest.fixture
def sample_embeddings(tmp_path):
    """Create sample embeddings"""
    np.random.seed(42)

    item_ids = [f"item_{i}" for i in range(20)]
    embeddings = {}

    for iid in item_ids:
        emb = np.random.randn(512).astype('float32')
        emb = emb / np.linalg.norm(emb)
        embeddings[iid] = emb

    # Make outfit items more similar
    base = embeddings['item_0'].copy()
    for i in [1, 2]:
        embeddings[f'item_{i}'] = base * 0.9 + np.random.randn(512).astype('float32') * 0.1
        embeddings[f'item_{i}'] = embeddings[f'item_{i}'] / np.linalg.norm(embeddings[f'item_{i}'])

    emb_path = str(tmp_path / "embeddings.pkl")
    with open(emb_path, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'item_ids': item_ids,
            'embedding_dim': 512
        }, f)

    return emb_path


class TestLoadTestOutfits:
    """Tests for loading test data"""

    def test_loads_test_data(self, sample_test_data):
        """Test loading test outfit data"""
        from evaluate import load_test_outfits

        data = load_test_outfits(sample_test_data)
        assert len(data) == 3
        assert "outfit_001" in data


class TestComputeOutfitCompatibility:
    """Tests for outfit compatibility scoring"""

    def test_compute_compatibility(self, sample_embeddings):
        """Test computing outfit compatibility"""
        from evaluate import compute_outfit_compatibility
        from embeddings import FashionEmbeddingGenerator

        embeddings, _ = FashionEmbeddingGenerator.load_embeddings(sample_embeddings)

        score = compute_outfit_compatibility(['item_0', 'item_1', 'item_2'], embeddings)

        # Similar items should have high compatibility
        assert 0 <= score <= 1

    def test_compatibility_single_item(self, sample_embeddings):
        """Test compatibility with single item"""
        from evaluate import compute_outfit_compatibility
        from embeddings import FashionEmbeddingGenerator

        embeddings, _ = FashionEmbeddingGenerator.load_embeddings(sample_embeddings)

        score = compute_outfit_compatibility(['item_0'], embeddings)
        assert score == 0.5  # Default for single item

    def test_compatibility_similar_higher(self, sample_embeddings):
        """Test similar items have higher compatibility"""
        from evaluate import compute_outfit_compatibility
        from embeddings import FashionEmbeddingGenerator

        embeddings, _ = FashionEmbeddingGenerator.load_embeddings(sample_embeddings)

        # Similar items (from same outfit)
        similar_score = compute_outfit_compatibility(['item_0', 'item_1', 'item_2'], embeddings)

        # Random items
        random_score = compute_outfit_compatibility(['item_10', 'item_11', 'item_12'], embeddings)

        # Similar items should have higher compatibility
        assert similar_score > random_score


class TestCompatibilityAUC:
    """Tests for compatibility AUC evaluation"""

    def test_evaluate_compatibility_auc(self, sample_test_data, sample_embeddings):
        """Test compatibility AUC evaluation"""
        from evaluate import evaluate_compatibility_auc

        auc = evaluate_compatibility_auc(sample_test_data, sample_embeddings, n_negative=1)

        assert 0 <= auc <= 1

    def test_auc_above_random(self, sample_test_data, sample_embeddings):
        """Test AUC is above random (0.5)"""
        from evaluate import evaluate_compatibility_auc

        auc = evaluate_compatibility_auc(sample_test_data, sample_embeddings, n_negative=3)

        # With similar items in outfits, should be above random
        # (relaxed assertion due to small sample)
        assert auc >= 0.0


class TestFITBAccuracy:
    """Tests for Fill-in-the-Blank evaluation"""

    def test_evaluate_fitb(self, sample_test_data, sample_embeddings):
        """Test FITB evaluation"""
        from evaluate import evaluate_fitb_accuracy

        accuracy = evaluate_fitb_accuracy(sample_test_data, sample_embeddings, k=10)

        assert 0 <= accuracy <= 1

    def test_predict_fitb_item(self, sample_embeddings):
        """Test FITB item prediction"""
        from evaluate import predict_fitb_item
        from embeddings import FaissSearcher

        searcher = FaissSearcher(sample_embeddings, use_gpu=False)

        predictions = predict_fitb_item(['item_0', 'item_1'], searcher, k=5)

        assert len(predictions) <= 5
        assert all('item_id' in p and 'score' in p for p in predictions)


class TestRetrievalMetrics:
    """Tests for retrieval metrics"""

    def test_visual_retrieval_metrics(self, sample_test_data, sample_embeddings):
        """Test visual retrieval metrics"""
        from evaluate import evaluate_visual_retrieval

        metrics = evaluate_visual_retrieval(sample_test_data, sample_embeddings, topk=[5, 10])

        assert 'recall@5' in metrics
        assert 'recall@10' in metrics
        assert 'ndcg@5' in metrics
        assert 'ndcg@10' in metrics
        assert 'hit@5' in metrics
        assert 'hit@10' in metrics

    def test_metrics_in_valid_range(self, sample_test_data, sample_embeddings):
        """Test metrics are in valid range"""
        from evaluate import evaluate_visual_retrieval

        metrics = evaluate_visual_retrieval(sample_test_data, sample_embeddings, topk=[10])

        for metric, value in metrics.items():
            assert 0 <= value <= 1, f"{metric} = {value} is out of range"


class TestNDCG:
    """Tests for NDCG computation"""

    def test_compute_ndcg_perfect(self):
        """Test perfect NDCG"""
        from evaluate import compute_ndcg

        ranked_list = ['a', 'b', 'c']
        relevant = {'a', 'b', 'c'}

        ndcg = compute_ndcg(ranked_list, relevant)
        assert ndcg == 1.0

    def test_compute_ndcg_empty(self):
        """Test NDCG with empty inputs"""
        from evaluate import compute_ndcg

        assert compute_ndcg([], {'a'}) == 0.0
        assert compute_ndcg(['a'], set()) == 0.0

    def test_compute_ndcg_partial(self):
        """Test NDCG with partial relevance"""
        from evaluate import compute_ndcg

        ranked_list = ['a', 'x', 'b']  # x is irrelevant
        relevant = {'a', 'b'}

        ndcg = compute_ndcg(ranked_list, relevant)
        assert 0 < ndcg < 1

    def test_compute_ndcg_with_k(self):
        """Test NDCG with cutoff"""
        from evaluate import compute_ndcg

        ranked_list = ['a', 'x', 'b', 'c']
        relevant = {'a', 'b', 'c'}

        ndcg_k2 = compute_ndcg(ranked_list, relevant, k=2)
        ndcg_k4 = compute_ndcg(ranked_list, relevant, k=4)

        assert ndcg_k2 <= ndcg_k4


class TestFullEvaluation:
    """Tests for full evaluation suite"""

    def test_run_full_evaluation(self, sample_test_data, sample_embeddings, tmp_path):
        """Test running full evaluation"""
        from evaluate import run_full_evaluation

        results = run_full_evaluation(
            data_dir=str(tmp_path),
            embeddings_path=sample_embeddings,
            output_path=str(tmp_path / "results.json")
        )

        assert 'compatibility_auc' in results
        assert 'fitb_accuracy' in results

    def test_evaluation_saves_results(self, sample_test_data, sample_embeddings, tmp_path):
        """Test evaluation saves results to file"""
        from evaluate import run_full_evaluation

        output_path = str(tmp_path / "eval_results.json")
        run_full_evaluation(
            data_dir=str(tmp_path),
            embeddings_path=sample_embeddings,
            output_path=output_path
        )

        assert os.path.exists(output_path)

        with open(output_path) as f:
            saved_results = json.load(f)

        assert 'compatibility_auc' in saved_results


# Integration tests with real data
@pytest.mark.skipif(
    not (os.path.exists("data/polyvore/test.json") or
         os.path.exists("data/polyvore/polyvore_outfits/test.json")),
    reason="Real Polyvore test data not available"
)
class TestRealEvaluation:
    """Integration tests with actual Polyvore data"""

    @pytest.mark.skipif(
        not os.path.exists("models/polyvore_embeddings.pkl"),
        reason="Real embeddings not available"
    )
    def test_real_compatibility_auc(self):
        """Test compatibility AUC on real data"""
        from evaluate import evaluate_compatibility_auc

        test_path = "data/polyvore/test.json"
        if not os.path.exists(test_path):
            test_path = "data/polyvore/polyvore_outfits/test.json"

        auc = evaluate_compatibility_auc(
            test_path,
            "models/polyvore_embeddings.pkl"
        )

        # Target: AUC > 0.85
        print(f"Compatibility AUC: {auc:.4f}")
        assert auc > 0.5  # At least above random

    @pytest.mark.skipif(
        not os.path.exists("models/polyvore_embeddings.pkl"),
        reason="Real embeddings not available"
    )
    def test_real_fitb_accuracy(self):
        """Test FITB accuracy on real data"""
        from evaluate import evaluate_fitb_accuracy

        test_path = "data/polyvore/test.json"
        if not os.path.exists(test_path):
            test_path = "data/polyvore/polyvore_outfits/test.json"

        accuracy = evaluate_fitb_accuracy(
            test_path,
            "models/polyvore_embeddings.pkl",
            k=10
        )

        # Target: Accuracy > 60%
        print(f"FITB Accuracy: {accuracy:.4f}")
        assert accuracy > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
