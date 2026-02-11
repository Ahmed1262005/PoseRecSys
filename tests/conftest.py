"""
Pytest configuration and shared fixtures for the recommendation system tests.
"""
import os
import sys
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))


# ============================================================================
# Fixtures: Test Data Factories
# ============================================================================

@pytest.fixture
def sample_candidate_dict() -> dict:
    """Sample candidate as dictionary (from Supabase query)."""
    return {
        "id": "test-sku-001",
        "sku_id": "test-sku-001",
        "name": "Test Cotton T-Shirt",
        "brand": "TestBrand",
        "price": 29.99,
        "category": "tops_woven",
        "article_type": "t-shirt",
        "image_url": "https://example.com/images/test-001.jpg",
        "similarity": 0.85,
        "product_attributes": {
            "occasions": ["casual", "everyday"],
            "pattern": "solid",
            "fit": "regular",
            "sleeve_length": "short",
            "color_family": "neutral"
        }
    }


@pytest.fixture
def sample_candidates_list(sample_candidate_dict: dict) -> list[dict]:
    """List of sample candidates for testing."""
    candidates = []
    for i in range(10):
        candidate = sample_candidate_dict.copy()
        candidate["id"] = f"test-sku-{i:03d}"
        candidate["sku_id"] = f"test-sku-{i:03d}"
        candidate["name"] = f"Test Item {i}"
        candidate["similarity"] = 0.95 - (i * 0.05)
        candidate["image_url"] = f"https://example.com/images/test-{i:03d}.jpg"
        candidates.append(candidate)
    return candidates


@pytest.fixture
def sample_user_profile() -> dict:
    """Sample user onboarding profile."""
    return {
        "user_id": "test-user-001",
        "core_setup": {
            "categories": ["tops_woven", "bottoms_trousers"],
            "sizes": {"top": "M", "bottom": "32"},
            "priceRange": {"min": 20, "max": 150}
        },
        "style_preferences": {
            "fits": ["regular", "slim"],
            "patterns": ["solid", "striped"],
            "colors": ["black", "white", "navy"]
        },
        "brand_preferences": {
            "preferred": ["Nike", "Adidas"],
            "avoided": ["Gucci"]
        }
    }


@pytest.fixture
def sample_taste_vector() -> list[float]:
    """Sample 512-dimensional taste vector."""
    import random
    random.seed(42)
    return [random.gauss(0, 1) for _ in range(512)]


# ============================================================================
# Fixtures: Mock Services
# ============================================================================

@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client for unit tests."""
    mock_client = MagicMock()
    
    # Mock RPC calls
    mock_client.rpc.return_value.execute.return_value.data = []
    
    # Mock table operations
    mock_client.table.return_value.select.return_value.execute.return_value.data = []
    mock_client.table.return_value.insert.return_value.execute.return_value.data = [{"id": "test"}]
    mock_client.table.return_value.upsert.return_value.execute.return_value.data = [{"id": "test"}]
    
    return mock_client


@pytest.fixture
def mock_supabase(mock_supabase_client):
    """Patch Supabase client creation."""
    with patch("supabase.create_client", return_value=mock_supabase_client):
        yield mock_supabase_client


# ============================================================================
# Fixtures: FastAPI Test Client
# ============================================================================

@pytest.fixture
def app():
    """Get the FastAPI application."""
    from api.app import create_app
    return create_app(include_static_files=False)


@pytest.fixture
async def async_client(app) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client for testing FastAPI endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ============================================================================
# Fixtures: Models
# ============================================================================

@pytest.fixture
def candidate_model():
    """Get the Candidate model class."""
    from recs.models import Candidate
    return Candidate


@pytest.fixture
def user_state_model():
    """Get the UserState model class."""
    from recs.models import UserState
    return UserState


@pytest.fixture
def sample_candidate(candidate_model, sample_candidate_dict):
    """Sample Candidate model instance."""
    from recs.filter_utils import candidate_from_dict
    return candidate_from_dict(sample_candidate_dict)


# ============================================================================
# Fixtures: Session State
# ============================================================================

@pytest.fixture
def in_memory_session_backend():
    """In-memory session state backend for testing."""
    from recs.session_state import InMemoryBackend
    return InMemoryBackend()


# ============================================================================
# Markers auto-use
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "supabase: marks tests that require Supabase")


# ============================================================================
# JWT Token Generation for Integration Tests
# ============================================================================

def generate_test_jwt(user_id: str = "test-user-001", exp_hours: int = 24) -> str:
    """
    Generate a test JWT token for integration tests.
    
    Args:
        user_id: The user ID to include in the token
        exp_hours: Hours until token expires (default 24)
        
    Returns:
        JWT token string
    """
    import jwt
    import time
    
    jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
    if not jwt_secret:
        raise ValueError("SUPABASE_JWT_SECRET environment variable required for integration tests")
    
    now = int(time.time())
    payload = {
        "sub": user_id,
        "aud": "authenticated",
        "role": "authenticated",
        "email": f"{user_id}@test.com",
        "aal": "aal1",
        "exp": now + (exp_hours * 3600),
        "iat": now,
        "is_anonymous": False,
    }
    
    return jwt.encode(payload, jwt_secret, algorithm="HS256")


@pytest.fixture
def test_jwt_token() -> str:
    """Fixture providing a valid test JWT token."""
    return generate_test_jwt()


@pytest.fixture
def auth_headers(test_jwt_token: str) -> dict:
    """Fixture providing auth headers with Bearer token."""
    return {"Authorization": f"Bearer {test_jwt_token}"}


# ============================================================================
# Skip conditions
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """Auto-skip integration tests if no server URL is configured."""
    skip_integration = pytest.mark.skip(reason="Integration tests require running server")
    skip_supabase = pytest.mark.skip(reason="Supabase tests require credentials")
    
    server_url = os.getenv("TEST_SERVER_URL")
    supabase_url = os.getenv("SUPABASE_URL")
    
    for item in items:
        if "integration" in item.keywords and not server_url:
            item.add_marker(skip_integration)
        if "supabase" in item.keywords and not supabase_url:
            item.add_marker(skip_supabase)
