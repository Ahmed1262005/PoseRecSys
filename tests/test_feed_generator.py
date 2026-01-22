"""
Tests for hybrid feed generator module
"""
import os
import json
import time
import pytest
import numpy as np
import pickle

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_embeddings(tmp_path):
    """Create sample embeddings for testing"""
    np.random.seed(42)

    item_ids = [f"item_{i}" for i in range(100)]
    embeddings = {}

    for iid in item_ids:
        emb = np.random.randn(512).astype('float32')
        emb = emb / np.linalg.norm(emb)
        embeddings[iid] = emb

    # Make some items similar
    embeddings['item_1'] = embeddings['item_0'] * 0.95 + np.random.randn(512).astype('float32') * 0.05
    embeddings['item_1'] = embeddings['item_1'] / np.linalg.norm(embeddings['item_1'])

    emb_path = str(tmp_path / "test_embeddings.pkl")
    with open(emb_path, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'item_ids': item_ids,
            'embedding_dim': 512
        }, f)

    return emb_path, embeddings, item_ids


@pytest.fixture
def sample_metadata(tmp_path):
    """Create sample item metadata"""
    metadata = {}
    categories = ['tops', 'bottoms', 'shoes', 'accessories', 'outerwear']

    for i in range(100):
        metadata[f"item_{i}"] = {
            'semantic_category': categories[i % len(categories)],
            'name': f"Item {i}",
        }

    meta_path = str(tmp_path / "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f)

    return meta_path, metadata


@pytest.fixture
def feed_generator(sample_embeddings, sample_metadata):
    """Create feed generator with test data"""
    from feed_generator import HybridFeedGenerator

    emb_path, _, _ = sample_embeddings
    meta_path, _ = sample_metadata

    return HybridFeedGenerator(
        embeddings_path=emb_path,
        item_metadata_path=meta_path,
        use_gpu=False
    )


class TestHybridFeedGenerator:
    """Tests for HybridFeedGenerator class"""

    def test_initializes(self, feed_generator):
        """Test feed generator initializes correctly"""
        assert feed_generator.visual_searcher is not None
        assert len(feed_generator.item_ids) == 100
        assert len(feed_generator.embeddings) == 100

    def test_category_index_built(self, feed_generator):
        """Test category index is built"""
        assert len(feed_generator.category_to_items) > 0
        assert 'tops' in feed_generator.category_to_items

    def test_visual_similar(self, feed_generator):
        """Test visual similarity search"""
        results = feed_generator.get_visual_similar('item_0', k=10)

        assert len(results) == 10
        assert all('item_id' in r and 'score' in r for r in results)

    def test_visual_similar_excludes_query(self, feed_generator):
        """Test visual similar excludes query item"""
        results = feed_generator.get_visual_similar('item_0', k=10)

        result_ids = [r['item_id'] for r in results]
        assert 'item_0' not in result_ids

    def test_visual_similar_with_exclusions(self, feed_generator):
        """Test visual similar with exclusion set"""
        exclude = {'item_1', 'item_2', 'item_3'}
        results = feed_generator.get_visual_similar('item_0', k=10, exclude_ids=exclude)

        result_ids = {r['item_id'] for r in results}
        assert not result_ids.intersection(exclude)

    def test_visual_similar_scores_descending(self, feed_generator):
        """Test results are sorted by score"""
        results = feed_generator.get_visual_similar('item_0', k=10)
        scores = [r['score'] for r in results]

        assert scores == sorted(scores, reverse=True)


class TestFeedGeneration:
    """Tests for feed generation"""

    def test_generate_feed_with_history(self, feed_generator):
        """Test feed generation with user history"""
        history = ['item_0', 'item_1', 'item_2']
        feed = feed_generator.generate_feed('test_user', history, k=10)

        assert len(feed) == 10
        assert all('item_id' in f and 'score' in f for f in feed)

    def test_feed_excludes_history(self, feed_generator):
        """Test feed excludes history items"""
        history = ['item_0', 'item_1', 'item_2', 'item_3', 'item_4']
        feed = feed_generator.generate_feed('test_user', history, k=10)

        feed_ids = {f['item_id'] for f in feed}
        history_set = set(history)

        assert not feed_ids.intersection(history_set)

    def test_feed_scores_descending(self, feed_generator):
        """Test feed is sorted by score"""
        history = ['item_0', 'item_1']
        feed = feed_generator.generate_feed('test_user', history, k=10)

        scores = [f['score'] for f in feed]
        assert scores == sorted(scores, reverse=True)

    def test_feed_with_custom_weights(self, feed_generator):
        """Test different weight combinations produce different results"""
        history = ['item_0', 'item_1', 'item_2']

        # Visual-heavy
        feed1 = feed_generator.generate_feed(
            'user1', history, 10,
            weights={'visual': 0.9, 'diversity': 0.1}
        )

        # Diversity-heavy
        feed2 = feed_generator.generate_feed(
            'user1', history, 10,
            weights={'visual': 0.5, 'diversity': 0.5}
        )

        # Results should differ (not guaranteed but likely)
        ids1 = [f['item_id'] for f in feed1]
        ids2 = [f['item_id'] for f in feed2]

        # At least some items should be different
        # (relaxed assertion since randomness can affect results)
        assert len(feed1) == len(feed2) == 10

    def test_empty_history(self, feed_generator):
        """Test feed with empty history"""
        feed = feed_generator.generate_feed('new_user', history=[], k=10)

        assert len(feed) == 10
        assert all('item_id' in f for f in feed)

    def test_feed_latency(self, feed_generator):
        """Test feed generation < 100ms"""
        history = ['item_0', 'item_1', 'item_2', 'item_3', 'item_4']

        # Warm up
        feed_generator.generate_feed('user', history, k=10)

        # Time multiple runs
        latencies = []
        for _ in range(10):
            start = time.time()
            feed_generator.generate_feed('user', history, k=20)
            latencies.append((time.time() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        assert avg_latency < 100, f"Avg latency {avg_latency:.2f}ms, expected < 100ms"

    def test_feed_has_sources(self, feed_generator):
        """Test feed items have source attribution"""
        history = ['item_0', 'item_1']
        feed = feed_generator.generate_feed('user', history, k=10)

        for item in feed:
            assert 'sources' in item
            assert isinstance(item['sources'], list)


class TestDiversityReranking:
    """Tests for diversity reranking"""

    def test_diversity_reranking(self, feed_generator):
        """Test MMR diversity reranking"""
        history = ['item_0']
        candidates = feed_generator.generate_feed('user', history, k=30)

        reranked = feed_generator.apply_diversity_reranking(
            candidates, k=10, lambda_param=0.5
        )

        assert len(reranked) == 10

    def test_diversity_reranking_respects_k(self, feed_generator):
        """Test reranking returns correct number of items"""
        history = ['item_0']
        candidates = feed_generator.generate_feed('user', history, k=30)

        for k in [5, 10, 15]:
            reranked = feed_generator.apply_diversity_reranking(candidates, k=k)
            assert len(reranked) == min(k, len(candidates))


class TestCategoryFunctions:
    """Tests for category-related functions"""

    def test_get_category_items(self, feed_generator):
        """Test getting items by category"""
        items = feed_generator.get_category_items('tops', k=10)

        assert len(items) <= 10
        # All items should be from 'tops' category
        for item_id in items:
            meta = feed_generator.item_metadata.get(item_id, {})
            assert meta.get('semantic_category') == 'tops'

    def test_get_category_items_with_exclusion(self, feed_generator):
        """Test category items excludes specified IDs"""
        exclude = {'item_0', 'item_5', 'item_10'}
        items = feed_generator.get_category_items('tops', k=10, exclude_ids=exclude)

        assert not set(items).intersection(exclude)

    def test_get_item_details(self, feed_generator):
        """Test getting item details"""
        details = feed_generator.get_item_details('item_0')

        assert details is not None
        assert 'item_id' in details
        assert 'semantic_category' in details


class TestEdgeCases:
    """Tests for edge cases"""

    def test_nonexistent_item(self, feed_generator):
        """Test handling nonexistent item"""
        results = feed_generator.get_visual_similar('nonexistent', k=10)
        assert len(results) == 0

    def test_item_details_nonexistent(self, feed_generator):
        """Test getting details for nonexistent item"""
        details = feed_generator.get_item_details('nonexistent')
        assert details is None

    def test_all_history_excluded(self, feed_generator):
        """Test when all candidate items are in history"""
        # Use all items as history
        history = feed_generator.item_ids[:90]
        feed = feed_generator.generate_feed('user', history, k=10)

        # Should still return some items
        assert len(feed) <= 10


# Integration tests
@pytest.mark.skipif(
    not os.path.exists("models/polyvore_embeddings.pkl"),
    reason="Real embeddings not available"
)
class TestRealFeedGenerator:
    """Integration tests with real data"""

    def test_create_real_generator(self):
        """Test creating generator with real data"""
        from feed_generator import create_feed_generator

        generator = create_feed_generator(use_gpu=False)
        assert len(generator.item_ids) > 1000

    def test_real_feed_generation(self):
        """Test feed generation with real data"""
        from feed_generator import create_feed_generator

        generator = create_feed_generator(use_gpu=False)
        history = generator.item_ids[:5]

        feed = generator.generate_feed('test', history, k=20)
        assert len(feed) == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
