"""
Tests for LocalVectorStore (FAISS-based semantic search).

Run with:
    PYTHONPATH=src python -m pytest tests/unit/test_local_vector_store.py -v

Covers:
    1. Index build + search with synthetic data
    2. Filtering (brand inclusion/exclusion, price range, exclude_ids)
    3. Adaptive over-fetch (retry on low survival)
    4. Result format matches _search_multimodal() output
    5. Snapshot save/load round-trip
    6. Singleton pattern
    7. Edge cases (empty index, zero results, all filtered)
    8. Search performance (< 5ms for 10K vectors)
"""

import os
import time
import tempfile

import numpy as np
import pytest

# Reset singleton before tests
import importlib


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset LocalVectorStore singleton between tests."""
    from search.local_vector_store import LocalVectorStore
    LocalVectorStore._instance = None
    yield
    LocalVectorStore._instance = None


def _build_test_store(
    n_vectors: int = 1000,
    dim: int = 512,
    n_brands: int = 10,
    price_range: tuple = (10.0, 200.0),
):
    """Build a LocalVectorStore with synthetic data for testing."""
    import faiss
    from search.local_vector_store import get_local_vector_store

    store = get_local_vector_store()

    # Generate random embeddings (L2-normalized)
    rng = np.random.RandomState(42)
    embeddings = rng.randn(n_vectors, dim).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= norms

    # Build FAISS index
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Generate product IDs and metadata
    brands = [f"Brand_{i}" for i in range(n_brands)]
    product_ids = [f"prod_{i:06d}" for i in range(n_vectors)]
    metadata = {}
    for i, pid in enumerate(product_ids):
        price = price_range[0] + (price_range[1] - price_range[0]) * (i / n_vectors)
        metadata[pid] = {
            "name": f"Product {i}",
            "brand": brands[i % n_brands],
            "price": round(price, 2),
            "original_price": round(price * 1.2, 2) if i % 3 == 0 else None,
            "in_stock": True,
            "image_url": f"https://img.test/{pid}.jpg",
            "gallery_images": [],
            "category": "Dresses" if i % 4 == 0 else "Tops",
            "broad_category": None,
            "article_type": "dress" if i % 4 == 0 else "top",
            "colors": ["black"] if i % 2 == 0 else ["red"],
            "materials": ["cotton"],
        }

    store._index = index
    store._product_ids = product_ids
    store._metadata = metadata
    store._id_to_row = {pid: i for i, pid in enumerate(product_ids)}
    store._count = n_vectors
    store._ready = True

    return store, embeddings


# =============================================================================
# 1. Basic Search
# =============================================================================

class TestBasicSearch:

    def test_search_returns_results(self):
        store, embs = _build_test_store()
        query = embs[0]  # search for first vector — should match itself
        results = store.search(query, limit=10)
        assert len(results) == 10

    def test_search_top_result_is_exact_match(self):
        store, embs = _build_test_store()
        query = embs[42]
        results = store.search(query, limit=5)
        assert results[0]["product_id"] == "prod_000042"
        assert results[0]["semantic_score"] > 0.99  # near-perfect match

    def test_search_respects_limit(self):
        store, embs = _build_test_store()
        for limit in [1, 5, 50, 100]:
            results = store.search(embs[0], limit=limit)
            assert len(results) == limit

    def test_search_scores_descending(self):
        store, embs = _build_test_store()
        results = store.search(embs[0], limit=50)
        scores = [r["semantic_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_not_ready_returns_empty(self):
        from search.local_vector_store import get_local_vector_store
        store = get_local_vector_store()
        # Not loaded yet
        results = store.search(np.zeros(512, dtype=np.float32), limit=10)
        assert results == []


# =============================================================================
# 2. Filtering
# =============================================================================

class TestFiltering:

    def test_include_brands(self):
        store, embs = _build_test_store()
        results = store.search(embs[0], limit=50, include_brands=["Brand_0"])
        assert all(r["brand"] == "Brand_0" for r in results)

    def test_include_brands_case_insensitive(self):
        store, embs = _build_test_store()
        results = store.search(embs[0], limit=50, include_brands=["brand_0"])
        assert all(r["brand"] == "Brand_0" for r in results)

    def test_exclude_brands(self):
        store, embs = _build_test_store()
        results = store.search(embs[0], limit=50, exclude_brands=["Brand_0", "Brand_1"])
        brands = {r["brand"] for r in results}
        assert "Brand_0" not in brands
        assert "Brand_1" not in brands

    def test_min_price(self):
        store, embs = _build_test_store()
        results = store.search(embs[0], limit=50, min_price=100.0)
        assert all(r["price"] >= 100.0 for r in results)

    def test_max_price(self):
        store, embs = _build_test_store()
        results = store.search(embs[0], limit=50, max_price=50.0)
        assert all(r["price"] <= 50.0 for r in results)

    def test_price_range(self):
        store, embs = _build_test_store()
        results = store.search(embs[0], limit=50, min_price=50.0, max_price=100.0)
        assert all(50.0 <= r["price"] <= 100.0 for r in results)

    def test_exclude_product_ids(self):
        store, embs = _build_test_store()
        exclude = {"prod_000000", "prod_000001", "prod_000002"}
        results = store.search(embs[0], limit=50, exclude_product_ids=exclude)
        result_ids = {r["product_id"] for r in results}
        assert result_ids.isdisjoint(exclude)

    def test_combined_filters(self):
        store, embs = _build_test_store()
        results = store.search(
            embs[0], limit=50,
            include_brands=["Brand_0"],
            min_price=50.0,
            max_price=150.0,
        )
        for r in results:
            assert r["brand"] == "Brand_0"
            assert 50.0 <= r["price"] <= 150.0

    def test_filter_returns_zero_when_impossible(self):
        store, embs = _build_test_store()
        # No product has price > 1000
        results = store.search(embs[0], limit=50, min_price=1000.0)
        assert len(results) == 0

    def test_multiple_include_brands(self):
        store, embs = _build_test_store()
        results = store.search(embs[0], limit=50, include_brands=["Brand_0", "Brand_1"])
        brands = {r["brand"] for r in results}
        assert brands.issubset({"Brand_0", "Brand_1"})


# =============================================================================
# 3. Result Format
# =============================================================================

class TestResultFormat:

    def test_result_has_all_required_fields(self):
        """Result dict must match _search_multimodal() output format."""
        store, embs = _build_test_store()
        results = store.search(embs[0], limit=1)
        r = results[0]

        required_fields = [
            "product_id", "name", "brand", "image_url", "gallery_images",
            "price", "original_price", "is_on_sale",
            "category_l1", "category_l2", "broad_category", "article_type",
            "primary_color", "color_family", "pattern", "apparent_fabric",
            "fit_type", "formality", "silhouette", "length", "neckline",
            "sleeve_type", "rise", "style_tags", "occasions", "seasons",
            "colors", "materials", "semantic_score", "source",
        ]
        for field in required_fields:
            assert field in r, f"Missing field: {field}"

    def test_source_is_semantic(self):
        store, embs = _build_test_store()
        results = store.search(embs[0], limit=5)
        assert all(r["source"] == "semantic" for r in results)

    def test_is_on_sale_computed_correctly(self):
        store, embs = _build_test_store()
        results = store.search(embs[0], limit=100)
        for r in results:
            if r["original_price"] is not None and r["original_price"] > r["price"]:
                assert r["is_on_sale"] is True
            else:
                assert r["is_on_sale"] is False

    def test_enrichable_fields_are_none(self):
        """Fields enriched from Algolia should be None (not populated by FAISS)."""
        store, embs = _build_test_store()
        r = store.search(embs[0], limit=1)[0]
        assert r["category_l1"] is None
        assert r["category_l2"] is None
        assert r["color_family"] is None
        assert r["apparent_fabric"] is None
        assert r["formality"] is None
        assert r["silhouette"] is None
        assert r["neckline"] is None
        assert r["rise"] is None


# =============================================================================
# 4. Snapshot Round-Trip
# =============================================================================

class TestSnapshotRoundTrip:

    def test_save_and_reload(self):
        store, embs = _build_test_store(n_vectors=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            store.save_snapshot(tmpdir)

            # Verify files exist
            assert os.path.exists(os.path.join(tmpdir, "index.faiss"))
            assert os.path.exists(os.path.join(tmpdir, "product_ids.npy"))
            assert os.path.exists(os.path.join(tmpdir, "metadata.pkl"))
            assert os.path.exists(os.path.join(tmpdir, "version.json"))

            # Reset and reload
            from search.local_vector_store import LocalVectorStore
            LocalVectorStore._instance = None
            store2 = LocalVectorStore()
            store2.load_snapshot(tmpdir)

            assert store2.ready
            assert store2.count == 100

            # Search should return same results
            r1 = store.search(embs[0], limit=5)
            r2 = store2.search(embs[0], limit=5)
            assert [r["product_id"] for r in r1] == [r["product_id"] for r in r2]

    def test_load_nonexistent_raises(self):
        from search.local_vector_store import get_local_vector_store
        store = get_local_vector_store()
        with pytest.raises(FileNotFoundError):
            store.load_snapshot("/nonexistent/path")


# =============================================================================
# 5. Singleton
# =============================================================================

class TestSingleton:

    def test_same_instance(self):
        from search.local_vector_store import get_local_vector_store
        a = get_local_vector_store()
        b = get_local_vector_store()
        assert a is b


# =============================================================================
# 6. Performance
# =============================================================================

class TestPerformance:

    def test_search_under_5ms_10k_vectors(self):
        store, embs = _build_test_store(n_vectors=10000)
        query = embs[0]

        # Warmup
        store.search(query, limit=100)

        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            store.search(query, limit=100)
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = sum(times) / len(times)
        print(f"  10K vectors search avg: {avg_ms:.2f}ms")
        assert avg_ms < 5.0, f"Search took {avg_ms:.2f}ms, expected < 5ms"

    def test_filtered_search_under_10ms_10k_vectors(self):
        store, embs = _build_test_store(n_vectors=10000)
        query = embs[0]

        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            store.search(
                query, limit=100,
                include_brands=["Brand_0"],
                min_price=50.0, max_price=150.0,
            )
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = sum(times) / len(times)
        print(f"  10K vectors filtered search avg: {avg_ms:.2f}ms")
        assert avg_ms < 10.0, f"Filtered search took {avg_ms:.2f}ms, expected < 10ms"

    def test_get_embedding_works(self):
        store, embs = _build_test_store(n_vectors=100)
        retrieved = store.get_embedding("prod_000042")
        assert retrieved is not None
        # Should be close to original (inner product ~ 1.0)
        similarity = np.dot(retrieved, embs[42])
        assert similarity > 0.99
