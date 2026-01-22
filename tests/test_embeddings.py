"""
Tests for embeddings module
"""
import os
import time
import tempfile
import pytest
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_image(tmp_path):
    """Create a sample test image"""
    img = Image.new('RGB', (224, 224), color='red')
    img_path = tmp_path / "sample.jpg"
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def sample_images(tmp_path):
    """Create multiple sample test images"""
    paths = []
    for i, color in enumerate(['red', 'green', 'blue', 'yellow']):
        img = Image.new('RGB', (224, 224), color=color)
        img_path = tmp_path / f"sample_{i}.jpg"
        img.save(img_path)
        paths.append(str(img_path))
    return paths


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing"""
    np.random.seed(42)
    item_ids = [f"item_{i}" for i in range(100)]
    embeddings = {iid: np.random.randn(512).astype('float32') for iid in item_ids}
    # Normalize
    for iid in embeddings:
        embeddings[iid] = embeddings[iid] / np.linalg.norm(embeddings[iid])
    return embeddings, item_ids


class TestFashionEmbeddingGenerator:
    """Tests for FashionEmbeddingGenerator class"""

    @pytest.mark.slow
    def test_model_loads(self):
        """Test FashionCLIP model loads correctly"""
        from embeddings import FashionEmbeddingGenerator

        gen = FashionEmbeddingGenerator()
        assert gen.model is not None
        assert gen.embedding_dim == 512

    @pytest.mark.slow
    def test_single_image_embedding(self, sample_image):
        """Test single image encoding"""
        from embeddings import FashionEmbeddingGenerator

        gen = FashionEmbeddingGenerator()
        embedding = gen.encode_image(sample_image)

        assert embedding.shape == (512,)
        assert np.isfinite(embedding).all()

    @pytest.mark.slow
    def test_embedding_normalization(self, sample_image):
        """Test embeddings are L2 normalized"""
        from embeddings import FashionEmbeddingGenerator

        gen = FashionEmbeddingGenerator()
        embedding = gen.encode_image(sample_image)

        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01, f"Norm is {norm}, expected ~1.0"

    @pytest.mark.slow
    def test_batch_embedding(self, sample_images):
        """Test batch encoding"""
        from embeddings import FashionEmbeddingGenerator

        gen = FashionEmbeddingGenerator()
        embeddings = gen.encode_batch(sample_images)

        assert embeddings.shape == (4, 512)
        assert np.isfinite(embeddings).all()

    @pytest.mark.slow
    def test_different_images_different_embeddings(self, sample_images):
        """Test different images produce different embeddings"""
        from embeddings import FashionEmbeddingGenerator

        gen = FashionEmbeddingGenerator()
        embeddings = gen.encode_batch(sample_images)

        # Embeddings should be different
        for i in range(len(sample_images)):
            for j in range(i + 1, len(sample_images)):
                similarity = np.dot(embeddings[i], embeddings[j])
                # Different colored images should have similarity < 1
                assert similarity < 0.99, f"Images {i} and {j} too similar: {similarity}"


class TestFaissIndex:
    """Tests for Faiss index building and search"""

    def test_faiss_index_creation(self, sample_embeddings):
        """Test Faiss index builds correctly"""
        from embeddings import FashionEmbeddingGenerator

        embeddings, item_ids = sample_embeddings

        gen = FashionEmbeddingGenerator.__new__(FashionEmbeddingGenerator)
        gen.embedding_dim = 512
        index = gen.build_faiss_index(embeddings, item_ids, use_gpu=False)

        assert index.ntotal == 100

    def test_faiss_search_returns_results(self, sample_embeddings):
        """Test Faiss search returns expected number of results"""
        from embeddings import FashionEmbeddingGenerator

        embeddings, item_ids = sample_embeddings

        gen = FashionEmbeddingGenerator.__new__(FashionEmbeddingGenerator)
        gen.embedding_dim = 512
        index = gen.build_faiss_index(embeddings, item_ids, use_gpu=False)

        # Search
        query = embeddings[item_ids[0]].reshape(1, -1).astype('float32')
        distances, indices = index.search(query, 10)

        assert len(indices[0]) == 10
        assert len(distances[0]) == 10

    def test_faiss_search_latency(self, sample_embeddings):
        """Test search is < 10ms"""
        from embeddings import FashionEmbeddingGenerator
        import faiss

        embeddings, item_ids = sample_embeddings

        # Build larger index for realistic test
        np.random.seed(42)
        large_embeddings = np.random.randn(10000, 512).astype('float32')
        faiss.normalize_L2(large_embeddings)

        index = faiss.IndexFlatIP(512)
        index.add(large_embeddings)

        query = np.random.randn(1, 512).astype('float32')
        faiss.normalize_L2(query)

        # Warm up
        index.search(query, 20)

        # Time search
        start = time.time()
        for _ in range(100):
            index.search(query, 20)
        elapsed = (time.time() - start) / 100 * 1000

        assert elapsed < 10, f"Search took {elapsed:.2f}ms, expected < 10ms"

    def test_similarity_correctness(self):
        """Test similar items have high scores"""
        import faiss

        # Create test embeddings where item 1 is similar to item 0
        embeddings = np.eye(100, 512).astype('float32')
        embeddings[1] = embeddings[0] * 0.99 + np.random.randn(512).astype('float32') * 0.01
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(512)
        index.add(embeddings)

        # Search for item 0
        distances, indices = index.search(embeddings[0:1], 5)

        # Item 1 should be in top results (besides item 0 itself)
        assert 1 in indices[0][:3], f"Item 1 not in top 3: {indices[0]}"


class TestEmbeddingsPersistence:
    """Tests for saving and loading embeddings"""

    def test_save_and_load_embeddings(self, sample_embeddings, tmp_path):
        """Test embeddings save and load correctly"""
        from embeddings import FashionEmbeddingGenerator

        embeddings, item_ids = sample_embeddings
        output_path = str(tmp_path / "test_embeddings.pkl")

        # Save
        gen = FashionEmbeddingGenerator.__new__(FashionEmbeddingGenerator)
        gen.embedding_dim = 512
        gen.save_embeddings(embeddings, item_ids, output_path)

        assert os.path.exists(output_path)

        # Load
        loaded_embeddings, loaded_ids = FashionEmbeddingGenerator.load_embeddings(output_path)

        assert len(loaded_embeddings) == len(embeddings)
        assert len(loaded_ids) == len(item_ids)
        assert np.allclose(loaded_embeddings['item_0'], embeddings['item_0'])

    def test_save_and_load_faiss_index(self, sample_embeddings, tmp_path):
        """Test Faiss index save and load correctly"""
        from embeddings import FashionEmbeddingGenerator

        embeddings, item_ids = sample_embeddings
        output_path = str(tmp_path / "test_index.bin")

        gen = FashionEmbeddingGenerator.__new__(FashionEmbeddingGenerator)
        gen.embedding_dim = 512

        # Build and save
        index = gen.build_faiss_index(embeddings, item_ids, use_gpu=False)
        gen.save_faiss_index(index, output_path)

        assert os.path.exists(output_path)

        # Load
        loaded_index = FashionEmbeddingGenerator.load_faiss_index(output_path, use_gpu=False)

        assert loaded_index.ntotal == index.ntotal


class TestFaissSearcher:
    """Tests for FaissSearcher class"""

    @pytest.fixture
    def searcher_setup(self, sample_embeddings, tmp_path):
        """Set up searcher with test data"""
        from embeddings import FashionEmbeddingGenerator, FaissSearcher

        embeddings, item_ids = sample_embeddings

        # Save embeddings
        emb_path = str(tmp_path / "embeddings.pkl")
        gen = FashionEmbeddingGenerator.__new__(FashionEmbeddingGenerator)
        gen.embedding_dim = 512
        gen.save_embeddings(embeddings, item_ids, emb_path)

        return emb_path, embeddings, item_ids

    def test_searcher_initializes(self, searcher_setup):
        """Test searcher initializes correctly"""
        from embeddings import FaissSearcher

        emb_path, embeddings, item_ids = searcher_setup
        searcher = FaissSearcher(emb_path, use_gpu=False)

        assert len(searcher.item_ids) == len(item_ids)
        assert searcher.index.ntotal == len(item_ids)

    def test_search_returns_valid_items(self, searcher_setup):
        """Test search returns valid items"""
        from embeddings import FaissSearcher

        emb_path, embeddings, item_ids = searcher_setup
        searcher = FaissSearcher(emb_path, use_gpu=False)

        results = searcher.search(item_ids[0], k=10)

        assert len(results) == 10
        assert all('item_id' in r and 'score' in r for r in results)

    def test_search_excludes_query(self, searcher_setup):
        """Test search excludes query item"""
        from embeddings import FaissSearcher

        emb_path, embeddings, item_ids = searcher_setup
        searcher = FaissSearcher(emb_path, use_gpu=False)

        query_id = item_ids[0]
        results = searcher.search(query_id, k=10, exclude_query=True)

        result_ids = [r['item_id'] for r in results]
        assert query_id not in result_ids

    def test_search_scores_descending(self, searcher_setup):
        """Test results are sorted by score descending"""
        from embeddings import FaissSearcher

        emb_path, embeddings, item_ids = searcher_setup
        searcher = FaissSearcher(emb_path, use_gpu=False)

        results = searcher.search(item_ids[0], k=10)
        scores = [r['score'] for r in results]

        assert scores == sorted(scores, reverse=True)

    def test_search_nonexistent_item(self, searcher_setup):
        """Test search returns empty for nonexistent item"""
        from embeddings import FaissSearcher

        emb_path, embeddings, item_ids = searcher_setup
        searcher = FaissSearcher(emb_path, use_gpu=False)

        results = searcher.search("nonexistent_item", k=10)

        assert len(results) == 0


# Integration tests with real data
@pytest.mark.skipif(
    not os.path.exists("data/polyvore/images"),
    reason="Real Polyvore images not available"
)
class TestRealPolyvoreEmbeddings:
    """Integration tests with actual Polyvore dataset"""

    @pytest.mark.slow
    def test_generate_sample_embeddings(self):
        """Test generating embeddings for sample of real images"""
        from embeddings import FashionEmbeddingGenerator
        import glob

        images = glob.glob("data/polyvore/images/*.jpg")[:10]

        if not images:
            pytest.skip("No images found")

        gen = FashionEmbeddingGenerator()
        embeddings = gen.encode_batch(images)

        assert embeddings.shape[0] == len(images)
        assert embeddings.shape[1] == 512


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
