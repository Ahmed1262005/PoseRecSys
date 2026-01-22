"""
Tests for FastAPI server
"""
import os
import json
import time
import pickle
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

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

    emb_path = str(tmp_path / "test_embeddings.pkl")
    with open(emb_path, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'item_ids': item_ids,
            'embedding_dim': 512
        }, f)

    return emb_path


@pytest.fixture
def sample_metadata(tmp_path):
    """Create sample item metadata"""
    metadata = {}
    categories = ['tops', 'bottoms', 'shoes', 'accessories']

    for i in range(100):
        metadata[f"item_{i}"] = {
            'semantic_category': categories[i % len(categories)],
            'name': f"Item {i}",
        }

    meta_path = str(tmp_path / "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f)

    return meta_path


@pytest.fixture
def test_client(sample_embeddings, sample_metadata, tmp_path):
    """Create test client with mocked generator"""
    from fastapi.testclient import TestClient

    # Set environment variables before importing api
    os.environ['EMBEDDINGS_PATH'] = sample_embeddings
    os.environ['METADATA_PATH'] = sample_metadata
    os.environ['USE_GPU'] = 'false'

    # Import and create app
    from api import app

    with TestClient(app) as client:
        yield client


class TestHealthEndpoint:
    """Tests for health check endpoint"""

    def test_health_check(self, test_client):
        """Test health endpoint returns 200"""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_health_includes_counts(self, test_client):
        """Test health includes item/category counts"""
        response = test_client.get("/health")
        data = response.json()

        assert "items_count" in data
        assert "categories_count" in data


class TestFeedEndpoint:
    """Tests for feed generation endpoint"""

    def test_feed_returns_200(self, test_client):
        """Test feed endpoint returns 200"""
        response = test_client.post("/feed", json={
            "user_id": "test_user",
            "k": 10
        })
        assert response.status_code == 200

    def test_feed_returns_correct_count(self, test_client):
        """Test feed returns requested number of items"""
        response = test_client.post("/feed", json={
            "user_id": "test_user",
            "k": 10
        })
        data = response.json()

        assert "items" in data
        assert len(data["items"]) == 10

    def test_feed_item_structure(self, test_client):
        """Test feed items have correct structure"""
        response = test_client.post("/feed", json={
            "user_id": "test_user",
            "k": 5
        })
        data = response.json()

        for item in data["items"]:
            assert "item_id" in item
            assert "score" in item

    def test_feed_with_custom_weights(self, test_client):
        """Test feed with custom weights"""
        response = test_client.post("/feed", json={
            "user_id": "test_user",
            "k": 10,
            "weights": {"visual": 0.8, "diversity": 0.2}
        })
        assert response.status_code == 200

    def test_feed_respects_k_limits(self, test_client):
        """Test feed respects k parameter limits"""
        # k > 100 should fail validation
        response = test_client.post("/feed", json={
            "user_id": "test_user",
            "k": 150
        })
        assert response.status_code == 422  # Validation error


class TestSimilarEndpoint:
    """Tests for similar items endpoint"""

    def test_similar_returns_200(self, test_client):
        """Test similar endpoint returns 200"""
        response = test_client.post("/similar", json={
            "item_id": "item_0",
            "k": 10
        })
        assert response.status_code == 200

    def test_similar_returns_correct_count(self, test_client):
        """Test similar returns requested count"""
        response = test_client.post("/similar", json={
            "item_id": "item_0",
            "k": 10
        })
        data = response.json()

        assert "similar" in data
        assert len(data["similar"]) == 10

    def test_similar_invalid_item(self, test_client):
        """Test similar returns 404 for invalid item"""
        response = test_client.post("/similar", json={
            "item_id": "nonexistent_item",
            "k": 10
        })
        assert response.status_code == 404


class TestFeedbackEndpoint:
    """Tests for feedback endpoint"""

    def test_feedback_returns_success(self, test_client):
        """Test feedback recording succeeds"""
        response = test_client.post("/feedback", json={
            "user_id": "test_user",
            "item_id": "item_0",
            "action": "view"
        })
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_feedback_updates_history(self, test_client):
        """Test feedback updates user history"""
        user_id = "history_test_user"

        # Record feedback
        test_client.post("/feedback", json={
            "user_id": user_id,
            "item_id": "item_0",
            "action": "view"
        })

        # Check history
        response = test_client.get(f"/user/{user_id}/history")
        data = response.json()

        assert "item_0" in data["history"]

    def test_feedback_invalid_item(self, test_client):
        """Test feedback with invalid item returns 404"""
        response = test_client.post("/feedback", json={
            "user_id": "test_user",
            "item_id": "nonexistent_item",
            "action": "view"
        })
        assert response.status_code == 404


class TestItemEndpoint:
    """Tests for item details endpoint"""

    def test_get_item_returns_200(self, test_client):
        """Test getting item details"""
        response = test_client.get("/item/item_0")
        assert response.status_code == 200

    def test_get_item_structure(self, test_client):
        """Test item response has correct structure"""
        response = test_client.get("/item/item_0")
        data = response.json()

        assert "item_id" in data
        assert "category" in data
        assert "has_embedding" in data

    def test_get_invalid_item(self, test_client):
        """Test getting nonexistent item returns 404"""
        response = test_client.get("/item/nonexistent")
        assert response.status_code == 404


class TestUserHistoryEndpoints:
    """Tests for user history endpoints"""

    def test_get_user_history(self, test_client):
        """Test getting user history"""
        response = test_client.get("/user/test_user/history")
        assert response.status_code == 200

        data = response.json()
        assert "history" in data
        assert "feedback_count" in data

    def test_clear_user_history(self, test_client):
        """Test clearing user history"""
        user_id = "clear_test_user"

        # Add to history
        test_client.post("/feedback", json={
            "user_id": user_id,
            "item_id": "item_0",
            "action": "view"
        })

        # Clear history
        response = test_client.delete(f"/user/{user_id}/history")
        assert response.status_code == 200

        # Verify cleared
        response = test_client.get(f"/user/{user_id}/history")
        assert len(response.json()["history"]) == 0


class TestCategoryEndpoints:
    """Tests for category endpoints"""

    def test_list_categories(self, test_client):
        """Test listing categories"""
        response = test_client.get("/categories")
        assert response.status_code == 200

        data = response.json()
        assert "categories" in data
        assert len(data["categories"]) > 0

    def test_get_category_items(self, test_client):
        """Test getting items by category"""
        response = test_client.get("/category/tops/items?k=5")
        assert response.status_code == 200

        data = response.json()
        assert "items" in data
        assert len(data["items"]) <= 5

    def test_get_invalid_category(self, test_client):
        """Test getting invalid category returns 404"""
        response = test_client.get("/category/invalid_category/items")
        assert response.status_code == 404


class TestStatsEndpoint:
    """Tests for stats endpoint"""

    def test_get_stats(self, test_client):
        """Test getting API stats"""
        response = test_client.get("/stats")
        assert response.status_code == 200

        data = response.json()
        assert "total_items" in data
        assert "total_categories" in data
        assert "generator_ready" in data


class TestLatency:
    """Tests for API latency requirements"""

    def test_feed_latency_p95(self, test_client):
        """Test feed endpoint P95 latency < 100ms"""
        latencies = []

        # Warm up
        test_client.post("/feed", json={"user_id": "perf", "k": 20})

        for _ in range(50):
            start = time.time()
            test_client.post("/feed", json={"user_id": "perf_test", "k": 20})
            latencies.append((time.time() - start) * 1000)

        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        assert p95 < 100, f"P95 latency {p95:.2f}ms, expected < 100ms"

    def test_similar_latency(self, test_client):
        """Test similar endpoint latency"""
        latencies = []

        for _ in range(20):
            start = time.time()
            test_client.post("/similar", json={"item_id": "item_0", "k": 20})
            latencies.append((time.time() - start) * 1000)

        avg = sum(latencies) / len(latencies)
        assert avg < 50, f"Avg latency {avg:.2f}ms, expected < 50ms"


class TestFeedWithHistory:
    """Tests for feed generation with user history"""

    def test_feed_uses_history(self, test_client):
        """Test feed is personalized based on history"""
        user_id = "history_user"

        # Record several interactions
        for i in range(5):
            test_client.post("/feedback", json={
                "user_id": user_id,
                "item_id": f"item_{i}",
                "action": "view"
            })

        # Get feed
        response = test_client.post("/feed", json={
            "user_id": user_id,
            "k": 10
        })
        data = response.json()

        # Feed should not contain history items
        history_ids = {f"item_{i}" for i in range(5)}
        feed_ids = {item["item_id"] for item in data["items"]}

        assert not feed_ids.intersection(history_ids)


# Integration tests
@pytest.mark.skipif(
    not os.path.exists("models/polyvore_embeddings.pkl"),
    reason="Real embeddings not available"
)
class TestRealAPI:
    """Integration tests with real data"""

    @pytest.fixture
    def real_client(self):
        """Create test client with real data"""
        from fastapi.testclient import TestClient
        from api import app

        with TestClient(app) as client:
            yield client

    def test_real_health_check(self, real_client):
        """Test health check with real data"""
        response = real_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_real_feed_generation(self, real_client):
        """Test feed generation with real data"""
        response = real_client.post("/feed", json={
            "user_id": "integration_test",
            "k": 20
        })
        assert response.status_code == 200
        assert len(response.json()["items"]) == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
