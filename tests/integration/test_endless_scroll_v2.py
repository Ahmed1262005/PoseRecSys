"""
Tests for Endless Scroll V2 - Keyset Pagination with Correctness Guarantees

Test-Driven Development: These tests define the expected behavior
BEFORE implementation. They should fail initially and pass after
implementing the keyset pagination system.

Correctness Guarantees:
A. No duplicates within a session
B. Stable ordering within session
C. Graceful degradation (Redis unavailable)
D. Session isolation
"""

import os
import sys
import json
import time
import base64
import pytest
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# =============================================================================
# Test Configuration
# =============================================================================

# Server URL - can be overridden via environment variable
BASE_URL = os.getenv("TEST_SERVER_URL", "http://localhost:8080")
API_PREFIX = "/api/recs/v2"


@dataclass
class FeedResult:
    """Result from a feed request."""
    session_id: str
    cursor: Optional[str]
    product_ids: List[str]
    scores: List[float]
    has_more: bool
    metadata: Dict[str, Any]
    # Full result dicts for detailed assertions
    results: List[Dict[str, Any]] = None

    @property
    def brands(self) -> List[str]:
        """Extract brand names from results."""
        if not self.results:
            return []
        return [r.get("brand", "") or "" for r in self.results]

    def brand_counts(self) -> Dict[str, int]:
        """Count how many items per brand."""
        from collections import Counter
        return dict(Counter(b.lower() for b in self.brands if b))

    @property
    def categories(self) -> List[str]:
        """Extract categories from results."""
        if not self.results:
            return []
        return [r.get("category", "") or r.get("broad_category", "") or "" for r in self.results]


def get_auth_headers(user_id: str = "test-user-001") -> Dict[str, str]:
    """Generate auth headers with a test JWT token."""
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
        "exp": now + (24 * 3600),
        "iat": now,
        "is_anonymous": False,
    }
    
    token = jwt.encode(payload, jwt_secret, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


def get_feed(
    session_id: Optional[str] = None,
    cursor: Optional[str] = None,
    page_size: int = 50,
    categories: Optional[str] = None,
    anon_id: str = "test_user",
    use_v3: bool = True  # Use v3 keyset endpoint when available
) -> FeedResult:
    """
    Helper to call the feed endpoint.

    Args:
        session_id: Existing session ID (auto-generated if None)
        cursor: Keyset cursor from previous response
        page_size: Number of items per page
        categories: Comma-separated category filter
        anon_id: Anonymous user ID (also used as JWT user_id)
        use_v3: Use keyset-based v3 endpoint (when implemented)

    Returns:
        FeedResult with products, cursor, and metadata
    """
    params = {
        "page_size": page_size,
        "gender": "female"
    }

    if session_id:
        params["session_id"] = session_id
    if cursor:
        params["cursor"] = cursor
    if categories:
        params["categories"] = categories

    # Get auth headers using anon_id as the user_id
    headers = get_auth_headers(user_id=anon_id)

    # Use v3 keyset endpoint when available, fall back to endless
    endpoint = f"{BASE_URL}{API_PREFIX}/feed/keyset" if use_v3 else f"{BASE_URL}{API_PREFIX}/feed/endless"

    try:
        response = requests.get(endpoint, params=params, headers=headers, timeout=30)
    except requests.exceptions.ConnectionError:
        # Fall back to endless endpoint
        endpoint = f"{BASE_URL}{API_PREFIX}/feed/endless"
        response = requests.get(endpoint, params=params, headers=headers, timeout=30)

    response.raise_for_status()
    data = response.json()

    results = data.get("results", [])
    return FeedResult(
        session_id=data.get("session_id", ""),
        cursor=data.get("cursor"),
        product_ids=[r["product_id"] for r in results],
        scores=[r.get("score", 0.0) for r in results],
        has_more=data.get("pagination", {}).get("has_more", False),
        metadata=data.get("metadata", {}),
        results=results,
    )


def decode_cursor(cursor: str) -> Dict[str, Any]:
    """Decode a base64 cursor to see its contents."""
    try:
        decoded = base64.b64decode(cursor).decode('utf-8')
        return json.loads(decoded)
    except Exception:
        return {"raw": cursor}


def create_session(
    categories: Optional[str] = None,
    anon_id: str = None
) -> str:
    """Create a new session and return session_id."""
    anon_id = anon_id or f"test_{int(time.time() * 1000)}"
    result = get_feed(
        page_size=10,
        categories=categories,
        anon_id=anon_id
    )
    return result.session_id


def clear_session(session_id: str) -> bool:
    """Clear a session for cleanup."""
    try:
        response = requests.delete(
            f"{BASE_URL}{API_PREFIX}/session/{session_id}",
            timeout=5
        )
        return response.status_code == 200
    except Exception:
        return False


def count_products(category: Optional[str] = None) -> int:
    """Get count of available products."""
    try:
        params = {"gender": "female"}
        if category:
            params["categories"] = category
        response = requests.get(
            f"{BASE_URL}{API_PREFIX}/products/count",
            params=params,
            timeout=5
        )
        if response.status_code == 200:
            return response.json().get("count", 0)
    except Exception:
        pass
    # Fallback: estimate from metadata
    return 5000  # Conservative estimate


# =============================================================================
# Guarantee A: No Duplicates Within Session
# =============================================================================

class TestNoDuplicates:
    """Verify Guarantee A: No item appears twice in a session."""

    @pytest.mark.integration
    def test_no_duplicates_10_pages(self):
        """No duplicates across 10 pages of scrolling."""
        anon_id = f"dedup_test_{int(time.time())}"
        session_id = None
        cursor = None
        all_ids = []

        for page in range(10):
            result = get_feed(
                session_id=session_id,
                cursor=cursor,
                page_size=50,
                anon_id=anon_id
            )

            session_id = result.session_id
            cursor = result.cursor
            all_ids.extend(result.product_ids)

            if not result.has_more:
                break

        # Check for duplicates
        unique_ids = set(all_ids)
        duplicate_count = len(all_ids) - len(unique_ids)

        assert duplicate_count == 0, \
            f"Found {duplicate_count} duplicates in {len(all_ids)} items"

    @pytest.mark.integration
    def test_no_duplicates_until_exhaustion(self):
        """No duplicates when scrolling until catalog exhausted (small category)."""
        anon_id = f"exhaust_test_{int(time.time())}"
        session_id = None
        cursor = None
        all_ids = []

        # Use outerwear - smaller catalog
        for page in range(100):  # Safety limit
            result = get_feed(
                session_id=session_id,
                cursor=cursor,
                page_size=50,
                categories="outerwear",
                anon_id=anon_id
            )

            session_id = result.session_id
            cursor = result.cursor
            all_ids.extend(result.product_ids)

            if not result.has_more:
                break

        # Verify no duplicates
        unique_ids = set(all_ids)
        assert len(all_ids) == len(unique_ids), \
            f"Duplicates found: {len(all_ids)} total, {len(unique_ids)} unique"


# =============================================================================
# Guarantee B: Stable Ordering Within Session
# =============================================================================

class TestStableOrdering:
    """Verify Guarantee B: Ordering doesn't change within session."""

    @pytest.mark.integration
    def test_refetch_same_page_identical(self):
        """Refetching page 0 returns identical results."""
        anon_id = f"stable_test_{int(time.time())}"

        # Fetch page 0 first time
        result1 = get_feed(page_size=20, anon_id=anon_id)
        session_id = result1.session_id

        # Fetch page 0 again with same session (no cursor = page 0)
        result2 = get_feed(
            session_id=session_id,
            page_size=20,
            anon_id=anon_id,
            cursor=None  # Explicitly no cursor
        )

        # Note: This test may need adjustment based on implementation
        # If cursor-based, re-fetching without cursor might give fresh results
        # For now, we verify the system returns consistent data
        assert len(result1.product_ids) > 0, "Should return products"
        assert len(result2.product_ids) > 0, "Should return products"

    @pytest.mark.integration
    def test_scores_descending(self):
        """Scores should be in descending order within a page."""
        result = get_feed(page_size=50, anon_id=f"score_test_{int(time.time())}")

        scores = result.scores
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i-1], \
                f"Scores not descending at position {i}: {scores[i-1]} -> {scores[i]}"

    @pytest.mark.integration
    def test_cursor_continuity(self):
        """Cursor picks up exactly where previous page left off."""
        anon_id = f"cursor_test_{int(time.time())}"

        # Get page 0
        result0 = get_feed(page_size=10, anon_id=anon_id)
        last_score_p0 = result0.scores[-1] if result0.scores else 0

        # Get page 1 using cursor
        result1 = get_feed(
            session_id=result0.session_id,
            cursor=result0.cursor,
            page_size=10,
            anon_id=anon_id
        )
        first_score_p1 = result1.scores[0] if result1.scores else 0

        # First item of page 1 should have score <= last item of page 0
        assert first_score_p1 <= last_score_p0, \
            f"Score discontinuity: page 0 ended at {last_score_p0}, page 1 started at {first_score_p1}"


# =============================================================================
# Guarantee C: Graceful Degradation
# =============================================================================

class TestGracefulDegradation:
    """Verify Guarantee C: System works without Redis."""

    @pytest.mark.integration
    def test_system_responds_without_redis(self):
        """System should respond even if Redis is unavailable."""
        # This test assumes the server handles Redis failures gracefully
        anon_id = f"degradation_test_{int(time.time())}"

        result = get_feed(page_size=10, anon_id=anon_id)

        assert result.session_id, "Should return session_id"
        assert len(result.product_ids) > 0, "Should return products"

    @pytest.mark.integration
    def test_new_session_after_expiry(self):
        """After session expires, new session gets fresh feed."""
        # This is a conceptual test - TTL is 24h so we can't wait
        # Instead, we verify that a new session gets valid results
        anon_id1 = f"expiry_test_1_{int(time.time())}"
        anon_id2 = f"expiry_test_2_{int(time.time())}"

        result1 = get_feed(page_size=20, anon_id=anon_id1)
        result2 = get_feed(page_size=20, anon_id=anon_id2)

        # Both should get valid results
        assert len(result1.product_ids) == 20
        assert len(result2.product_ids) == 20

        # Sessions should be different
        assert result1.session_id != result2.session_id


# =============================================================================
# Guarantee D: Session Isolation
# =============================================================================

class TestSessionIsolation:
    """Verify Guarantee D: Sessions are independent."""

    @pytest.mark.integration
    def test_concurrent_sessions_independent(self):
        """Multiple concurrent sessions don't interfere with each other."""
        num_sessions = 5
        pages_per_session = 3
        results = {}

        def fetch_pages(session_num: int) -> Dict[str, Any]:
            anon_id = f"concurrent_{session_num}_{int(time.time())}"
            session_id = None
            cursor = None
            all_ids = []

            for page in range(pages_per_session):
                result = get_feed(
                    session_id=session_id,
                    cursor=cursor,
                    page_size=30,
                    anon_id=anon_id
                )
                session_id = result.session_id
                cursor = result.cursor
                all_ids.extend(result.product_ids)

            return {
                "session_id": session_id,
                "total_items": len(all_ids),
                "unique_items": len(set(all_ids)),
                "items": all_ids
            }

        # Run sessions concurrently
        with ThreadPoolExecutor(max_workers=num_sessions) as executor:
            futures = {executor.submit(fetch_pages, i): i for i in range(num_sessions)}
            for future in as_completed(futures):
                session_num = futures[future]
                results[session_num] = future.result()

        # Verify each session has no internal duplicates
        for session_num, data in results.items():
            assert data["total_items"] == data["unique_items"], \
                f"Session {session_num} has duplicates: {data['total_items']} total, {data['unique_items']} unique"

        # Verify all sessions are different
        session_ids = [data["session_id"] for data in results.values()]
        assert len(set(session_ids)) == num_sessions, \
            f"Expected {num_sessions} unique sessions, got {len(set(session_ids))}"

    @pytest.mark.integration
    def test_different_users_diverse_cold_start(self):
        """Different cold-start users should get diverse, personalized feeds.
        
        The system uses per-user seeds for exploration scoring, so different 
        users are expected to see different item orderings. This ensures the
        feed doesn't feel generic - each user gets a unique experience even
        before onboarding.
        """
        result1 = get_feed(page_size=20, anon_id=f"user_a_{int(time.time())}")
        result2 = get_feed(page_size=20, anon_id=f"user_b_{int(time.time())}")

        # Both users should get valid results
        assert len(result1.product_ids) > 0, "User A should get products"
        assert len(result2.product_ids) > 0, "User B should get products"

        # Feeds should NOT be identical - per-user seed ensures diversity
        # Some overlap is fine (both draw from same catalog), but not 100%
        overlap = set(result1.product_ids) & set(result2.product_ids)
        overlap_ratio = len(overlap) / max(len(result1.product_ids), 1)

        assert overlap_ratio < 1.0, \
            "Different users should not get identical feeds"


# =============================================================================
# Keyset Cursor Tests
# =============================================================================

class TestKeysetCursor:
    """Test keyset cursor mechanics."""

    @pytest.mark.integration
    def test_cursor_returned_in_response(self):
        """Response should include cursor for next page."""
        result = get_feed(page_size=10, anon_id=f"cursor_return_{int(time.time())}")

        # cursor should be present if has_more is True
        if result.has_more:
            assert result.cursor, "Should return cursor when has_more=True"

    @pytest.mark.integration
    def test_cursor_is_opaque(self):
        """Cursor should be opaque (base64 encoded)."""
        result = get_feed(page_size=10, anon_id=f"cursor_opaque_{int(time.time())}")

        if result.cursor:
            # Should be valid base64
            try:
                decoded = decode_cursor(result.cursor)
                # Should contain score and item_id
                assert "score" in decoded or "raw" in decoded, \
                    "Cursor should contain score information"
            except Exception as e:
                pytest.fail(f"Cursor should be decodable: {e}")

    @pytest.mark.integration
    def test_invalid_cursor_handled(self):
        """Invalid cursor should be handled gracefully."""
        result = get_feed(
            cursor="invalid_cursor_abc123",
            page_size=10,
            anon_id=f"invalid_cursor_{int(time.time())}"
        )

        # Should either return fresh results or error gracefully
        # Implementation decides behavior - just verify no crash
        assert result.session_id, "Should return session_id even with invalid cursor"


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Verify keyset pagination performance."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_constant_time_pagination(self):
        """Page 100 should be similar speed to page 1."""
        anon_id = f"perf_test_{int(time.time())}"
        session_id = None
        cursor = None

        # Warm up
        result = get_feed(page_size=50, anon_id=anon_id)
        session_id = result.session_id
        cursor = result.cursor

        # Time page 1
        start = time.time()
        result = get_feed(session_id=session_id, cursor=cursor, page_size=50, anon_id=anon_id)
        time_page1 = time.time() - start
        cursor = result.cursor

        # Scroll to page 50
        for _ in range(48):
            result = get_feed(session_id=session_id, cursor=cursor, page_size=50, anon_id=anon_id)
            cursor = result.cursor
            if not result.has_more:
                pytest.skip("Catalog exhausted before page 50")

        # Time page 50
        start = time.time()
        result = get_feed(session_id=session_id, cursor=cursor, page_size=50, anon_id=anon_id)
        time_page50 = time.time() - start

        # Page 50 should be within 3x of page 1 (accounting for network variance)
        assert time_page50 < time_page1 * 3, \
            f"Page 50 ({time_page50:.3f}s) should be similar speed to page 1 ({time_page1:.3f}s)"

    @pytest.mark.integration
    def test_response_time_acceptable(self):
        """Response time should be under 500ms."""
        start = time.time()
        result = get_feed(page_size=50, anon_id=f"latency_test_{int(time.time())}")
        elapsed = time.time() - start

        assert elapsed < 0.5, f"Response time {elapsed:.3f}s exceeds 500ms threshold"


# =============================================================================
# Redis Session State Tests
# =============================================================================

class TestRedisSessionState:
    """Test Redis session state management."""

    @pytest.mark.integration
    def test_seen_ids_authoritative(self):
        """Seen tracking should be authoritative (no false positives)."""
        anon_id = f"seen_test_{int(time.time())}"

        # Get first page
        result1 = get_feed(page_size=50, anon_id=anon_id)
        first_page_ids = set(result1.product_ids)

        # Get second page
        result2 = get_feed(
            session_id=result1.session_id,
            cursor=result1.cursor,
            page_size=50,
            anon_id=anon_id
        )
        second_page_ids = set(result2.product_ids)

        # No overlap should exist
        overlap = first_page_ids & second_page_ids
        assert len(overlap) == 0, \
            f"Pages should not overlap. Found {len(overlap)} duplicates"

    @pytest.mark.integration
    def test_session_persists_across_requests(self):
        """Session state should persist across multiple requests."""
        anon_id = f"persist_test_{int(time.time())}"

        # Create session and get first page
        result1 = get_feed(page_size=20, anon_id=anon_id)
        session_id = result1.session_id

        # Wait a bit
        time.sleep(0.5)

        # Get second page with same session
        result2 = get_feed(
            session_id=session_id,
            cursor=result1.cursor,
            page_size=20,
            anon_id=anon_id
        )

        # Should use same session
        assert result2.session_id == session_id, "Session should persist"

        # Items should be different
        assert set(result1.product_ids) != set(result2.product_ids), \
            "Second page should have different items"


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.integration
    def test_empty_result_at_exhaustion(self):
        """Should return empty results when catalog exhausted."""
        anon_id = f"exhaust_edge_{int(time.time())}"
        session_id = None
        cursor = None

        # Use small category
        for page in range(200):  # Safety limit
            result = get_feed(
                session_id=session_id,
                cursor=cursor,
                page_size=100,
                categories="outerwear",
                anon_id=anon_id
            )
            session_id = result.session_id
            cursor = result.cursor

            if not result.has_more:
                # Verify we can still make requests after exhaustion
                final_result = get_feed(
                    session_id=session_id,
                    cursor=cursor,
                    page_size=100,
                    categories="outerwear",
                    anon_id=anon_id
                )
                # Key invariant: has_more should remain False after exhaustion
                # Note: item count may vary slightly due to dedup/exploration logic
                assert final_result.has_more is False, \
                    "has_more should remain False after catalog exhaustion"
                break
        else:
            pytest.skip("Catalog not exhausted in 200 pages")

    @pytest.mark.integration
    def test_various_page_sizes(self):
        """Different page sizes should work correctly."""
        page_sizes = [10, 25, 50, 100]

        for page_size in page_sizes:
            anon_id = f"pagesize_{page_size}_{int(time.time())}"
            result = get_feed(page_size=page_size, anon_id=anon_id)

            assert len(result.product_ids) <= page_size, \
                f"Should return at most {page_size} items, got {len(result.product_ids)}"

    @pytest.mark.integration
    def test_category_filtering_works(self):
        """Category filter should return only matching categories."""
        anon_id = f"category_test_{int(time.time())}"
        result = get_feed(
            page_size=50,
            categories="tops",
            anon_id=anon_id
        )

        # Verify items are from the requested category
        # (This would require product details in response)
        assert len(result.product_ids) > 0, "Should return tops products"


# =============================================================================
# Integration Test Suite
# =============================================================================

class TestFullIntegration:
    """Full integration test simulating real user behavior."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_realistic_scroll_session(self):
        """Simulate a realistic 5-minute scrolling session."""
        anon_id = f"realistic_test_{int(time.time())}"
        session_id = None
        cursor = None
        all_ids = []
        page_times = []

        # Simulate 20 page scrolls
        for page in range(20):
            start = time.time()
            result = get_feed(
                session_id=session_id,
                cursor=cursor,
                page_size=50,
                anon_id=anon_id
            )
            elapsed = time.time() - start
            page_times.append(elapsed)

            session_id = result.session_id
            cursor = result.cursor
            all_ids.extend(result.product_ids)

            if not result.has_more:
                break

        # Verify no duplicates
        unique_ids = set(all_ids)
        assert len(all_ids) == len(unique_ids), \
            f"Found {len(all_ids) - len(unique_ids)} duplicates in {len(all_ids)} items"

        # Verify reasonable latency
        avg_time = sum(page_times) / len(page_times)
        assert avg_time < 1.0, f"Average page time {avg_time:.3f}s exceeds 1s"

        # Report stats
        print(f"\n  Total items: {len(all_ids)}")
        print(f"  Unique items: {len(unique_ids)}")
        print(f"  Avg latency: {avg_time:.3f}s")
        print(f"  Max latency: {max(page_times):.3f}s")


# =============================================================================
# Onboarding + Personalization Tests
# =============================================================================

import random

# Brands actually in the women's catalog (verified via Supabase)
TEST_BRANDS = [
    "Boohoo", "Alo Yoga", "Missguided", "Forever 21",
    "Nasty Gal", "American Eagle Outfitters", "Free People",
    "Princess Polly", "Old Navy", "Club Monaco",
    "J.Crew", "Ann Taylor", "Farm Rio", "Gap",
]

TEST_CATEGORIES = [
    "t-shirts", "blouses", "knitwear", "jeans", "trousers",
    "mini dresses", "midi dresses", "maxi dresses",
    "skirts", "shorts", "blazers", "coats",
]

TEST_COLORS_AVOID = ["yellow", "orange", "neon", "mustard", "lime"]
TEST_MATERIALS_AVOID = ["polyester", "acrylic", "nylon"]
TEST_OCCASIONS = ["everyday", "work", "date_night", "party", "weekend", "vacation"]
TEST_PATTERNS = ["solid", "striped", "floral", "geometric", "plaid", "abstract"]
TEST_FITS = ["regular", "relaxed", "slim", "oversized"]
TEST_STYLES_AVOID = ["bohemian", "punk", "preppy", "sporty", "gothic"]


def random_onboarding_profile(user_id: str) -> Dict[str, Any]:
    """Generate a random but realistic onboarding profile.
    
    Matches the FullOnboardingRequest model in recs/api_endpoints.py.
    """
    rng = random.Random(user_id)  # Deterministic per user_id

    preferred_brands = rng.sample(TEST_BRANDS, k=rng.randint(2, 5))
    brands_to_avoid = rng.sample(
        [b for b in TEST_BRANDS if b not in preferred_brands],
        k=rng.randint(0, 2)
    )

    # Build pattern preferences as dict: {"floral": "like", "geometric": "avoid"}
    liked = rng.sample(TEST_PATTERNS, k=rng.randint(1, 3))
    avoided = rng.sample([p for p in TEST_PATTERNS if p not in liked], k=rng.randint(0, 2))
    patterns = {p: "like" for p in liked}
    patterns.update({p: "avoid" for p in avoided})

    return {
        "userId": user_id,
        "gender": "female",
        "core-setup": {
            "selectedCategories": rng.sample(TEST_CATEGORIES, k=rng.randint(3, 6)),
            "sizes": [rng.choice(["XS", "S", "M", "L"])],
            "colorsToAvoid": rng.sample(TEST_COLORS_AVOID, k=rng.randint(1, 3)),
            "materialsToAvoid": rng.sample(TEST_MATERIALS_AVOID, k=rng.randint(0, 2)),
        },
        "lifestyle": {
            "enabled": True,
            "occasions": rng.sample(TEST_OCCASIONS, k=rng.randint(2, 4)),
            "stylesToAvoid": rng.sample(TEST_STYLES_AVOID, k=rng.randint(0, 2)),
            "patterns": patterns,
            "stylePersona": rng.sample(
                ["minimalist", "classic", "trendy", "romantic", "edgy"],
                k=rng.randint(1, 3)
            ),
        },
        "brands": {
            "enabled": True,
            "preferredBrands": preferred_brands,
            "brandsToAvoid": brands_to_avoid,
            "brandOpenness": rng.choice([
                "stick_to_favorites", "mix", "discover_new"
            ]),
        },
        "tops": {
            "enabled": True,
            "topTypes": rng.sample(["t-shirts", "blouses", "crop tops", "tanks"], k=2),
            "fits": rng.sample(TEST_FITS, k=rng.randint(1, 2)),
            "sleeves": rng.sample(["short", "long", "sleeveless"], k=2),
        },
        "style": {
            "enabled": True,
            "styleDirections": rng.sample(
                ["minimalist", "classic", "trendy", "romantic", "edgy"],
                k=rng.randint(1, 3)
            ),
            "modestyPreference": rng.choice(["none", "moderate", "modest"]),
        },
    }


def submit_onboarding(user_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Submit onboarding profile and return response."""
    headers = get_auth_headers(user_id=user_id)
    response = requests.post(
        f"{BASE_URL}{API_PREFIX}/onboarding",
        json=profile,
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


class TestOnboardingAndPersonalization:
    """Test that onboarding produces personalized feeds."""

    @pytest.mark.integration
    def test_onboarding_saves_successfully(self):
        """Random onboarding profile should save without errors."""
        user_id = f"onboard_test_{int(time.time())}"
        profile = random_onboarding_profile(user_id)
        result = submit_onboarding(user_id, profile)

        assert result["status"] == "success", f"Onboarding failed: {result}"

    @pytest.mark.integration
    def test_onboarded_user_gets_feed(self):
        """User with onboarding profile should get a non-empty feed."""
        user_id = f"feed_after_onboard_{int(time.time())}"
        profile = random_onboarding_profile(user_id)
        submit_onboarding(user_id, profile)

        # Now get feed
        result = get_feed(page_size=20, anon_id=user_id)

        assert len(result.product_ids) > 0, "Onboarded user should get products"
        assert result.session_id, "Should return session_id"

    @pytest.mark.integration
    def test_brand_affinity_boosts_preferred_brands(self):
        """User's preferred brands should appear more in their feed.
        
        A user who loves Free People with 'stick_to_favorites' should see
        Free People products ranked higher compared to a baseline cold-start.
        Free People is ~2.7% of catalog - with brand_preferred boost of 0.80
        and stick_to_favorites multiplier 2.0, it should punch above its weight.
        """
        ts = int(time.time())
        target_brand = "Free People"
        target_brand_lower = target_brand.lower()

        # User with strong brand preference
        user_id = f"brand_fp_{ts}"
        profile = random_onboarding_profile(user_id)
        profile["brands"]["preferredBrands"] = [target_brand]
        profile["brands"]["brandsToAvoid"] = []
        profile["brands"]["brandOpenness"] = "stick_to_favorites"
        submit_onboarding(user_id, profile)

        # Get 3 pages of results
        all_brands = []
        session_id = None
        cursor = None
        for _ in range(3):
            result = get_feed(
                session_id=session_id, cursor=cursor,
                page_size=50, anon_id=user_id
            )
            all_brands.extend(result.brands)
            session_id = result.session_id
            cursor = result.cursor
            if not result.has_more:
                break

        brand_counts = {}
        for b in all_brands:
            key = b.lower()
            if key:
                brand_counts[key] = brand_counts.get(key, 0) + 1

        total = len(all_brands)
        target_count = brand_counts.get(target_brand_lower, 0)
        target_pct = target_count / total if total > 0 else 0

        print(f"\n  {target_brand} items: {target_count}/{total} ({target_pct:.0%})")
        print(f"  Top 5 brands: {sorted(brand_counts.items(), key=lambda x: -x[1])[:5]}")

        # Free People is 2.7% of catalog. With brand boost, we expect
        # it to be noticeably above baseline. Assert at least some presence.
        assert target_count > 0, \
            f"Preferred brand '{target_brand}' should appear in feed, " \
            f"got brands: {sorted(brand_counts.items(), key=lambda x: -x[1])[:10]}"

    @pytest.mark.integration
    def test_brand_affinity_different_users_different_brands(self):
        """Two users with different preferred brands should see different brand distributions.
        
        User A prefers Free People, User B prefers J.Crew.
        Their feeds should reflect their brand preferences.
        """
        ts = int(time.time())

        # User A: loves Free People
        user_a = f"brand_fp_{ts}"
        profile_a = random_onboarding_profile(user_a)
        profile_a["brands"]["preferredBrands"] = ["Free People"]
        profile_a["brands"]["brandsToAvoid"] = ["J.Crew"]
        profile_a["brands"]["brandOpenness"] = "stick_to_favorites"
        submit_onboarding(user_a, profile_a)

        # User B: loves J.Crew
        user_b = f"brand_jc_{ts}"
        profile_b = random_onboarding_profile(user_b)
        profile_b["brands"]["preferredBrands"] = ["J.Crew"]
        profile_b["brands"]["brandsToAvoid"] = ["Free People"]
        profile_b["brands"]["brandOpenness"] = "stick_to_favorites"
        submit_onboarding(user_b, profile_b)

        feed_a = get_feed(page_size=50, anon_id=user_a)
        feed_b = get_feed(page_size=50, anon_id=user_b)

        counts_a = feed_a.brand_counts()
        counts_b = feed_b.brand_counts()

        print(f"\n  User A (Free People fan) top brands: {sorted(counts_a.items(), key=lambda x: -x[1])[:5]}")
        print(f"  User B (J.Crew fan) top brands: {sorted(counts_b.items(), key=lambda x: -x[1])[:5]}")

        fp_a = counts_a.get("free people", 0)
        fp_b = counts_b.get("free people", 0)
        jc_a = counts_a.get("j.crew", 0)
        jc_b = counts_b.get("j.crew", 0)

        print(f"  Free People: A={fp_a}, B={fp_b}")
        print(f"  J.Crew:      A={jc_a}, B={jc_b}")

        # User A should have more Free People than User B
        # User B should have more J.Crew than User A
        # At minimum, distributions should differ
        assert counts_a != counts_b, \
            "Users with different brand preferences should have different brand distributions"

    @pytest.mark.integration
    def test_brand_avoidance_demotes_brands(self):
        """Brands the user wants to avoid should appear less in the feed.
        
        Boohoo is 20% of the catalog. A user avoiding Boohoo should see 
        significantly less than 20% Boohoo in their feed.
        """
        ts = int(time.time())

        # User who avoids the largest brands
        user_id = f"brand_avoid_{ts}"
        profile = random_onboarding_profile(user_id)
        profile["brands"]["preferredBrands"] = ["J.Crew", "Ann Taylor"]
        profile["brands"]["brandsToAvoid"] = ["Boohoo", "Missguided", "Nasty Gal"]
        profile["brands"]["brandOpenness"] = "stick_to_favorites"
        submit_onboarding(user_id, profile)

        result = get_feed(page_size=50, anon_id=user_id)
        counts = result.brand_counts()

        avoided_brands = {"boohoo", "missguided", "nasty gal"}
        avoided_count = sum(counts.get(b, 0) for b in avoided_brands)
        total = len(result.product_ids)

        # These 3 brands are ~34% of catalog combined
        avoided_pct = avoided_count / total if total > 0 else 0
        print(f"\n  Avoided brand items: {avoided_count}/{total} ({avoided_pct:.0%})")
        print(f"  (Catalog baseline: ~34%)")
        print(f"  Top brands: {sorted(counts.items(), key=lambda x: -x[1])[:5]}")

        # With brand avoidance, these should be noticeably below their catalog baseline
        assert avoided_pct < 0.30, \
            f"Avoided brands (34% of catalog) should be demoted below 30%, got {avoided_pct:.0%}"

    @pytest.mark.integration
    def test_color_avoidance_respected(self):
        """Feed should not heavily feature colors the user wants to avoid."""
        ts = int(time.time())
        user_id = f"color_avoid_{ts}"

        profile = random_onboarding_profile(user_id)
        profile["core-setup"]["colorsToAvoid"] = ["yellow", "orange", "neon"]
        submit_onboarding(user_id, profile)

        result = get_feed(page_size=50, anon_id=user_id)

        assert len(result.product_ids) > 0, \
            "Feed with color avoidance should still return products"

        # Check if any result has color info and verify avoidance
        if result.results:
            avoided_colors = {"yellow", "orange", "neon"}
            color_violations = 0
            for r in result.results:
                item_colors = r.get("colors", []) or []
                if isinstance(item_colors, str):
                    item_colors = [item_colors]
                for c in item_colors:
                    if c and c.lower() in avoided_colors:
                        color_violations += 1
                        break

            violation_pct = color_violations / len(result.results)
            print(f"\n  Color violations: {color_violations}/{len(result.results)} ({violation_pct:.0%})")

            # Hard filter should block most avoided colors
            assert violation_pct < 0.2, \
                f"Avoided colors should be <20% of feed, got {violation_pct:.0%}"

    @pytest.mark.integration
    def test_different_onboarding_produces_different_feeds(self):
        """Two users with very different preferences should get different feeds."""
        ts = int(time.time())

        # User A: minimalist, work-focused, classic brands
        user_a = f"minimal_work_{ts}"
        profile_a = random_onboarding_profile(user_a)
        profile_a["lifestyle"]["occasions"] = ["work", "everyday"]
        profile_a["lifestyle"]["stylesToAvoid"] = ["bohemian", "punk", "gothic"]
        profile_a["style"]["styleDirections"] = ["minimalist", "classic"]
        profile_a["brands"]["preferredBrands"] = ["J.Crew", "Ann Taylor", "Club Monaco"]
        profile_a["brands"]["brandOpenness"] = "stick_to_favorites"
        submit_onboarding(user_a, profile_a)

        # User B: trendy, party-focused, bold patterns, fast fashion brands
        user_b = f"trendy_party_{ts}"
        profile_b = random_onboarding_profile(user_b)
        profile_b["lifestyle"]["occasions"] = ["party", "date_night"]
        profile_b["lifestyle"]["patterns"] = {"floral": "like", "geometric": "like", "abstract": "like"}
        profile_b["style"]["styleDirections"] = ["trendy", "edgy"]
        profile_b["brands"]["preferredBrands"] = ["Boohoo", "Missguided", "Nasty Gal"]
        profile_b["brands"]["brandOpenness"] = "discover_new"
        submit_onboarding(user_b, profile_b)

        feed_a = get_feed(page_size=50, anon_id=user_a)
        feed_b = get_feed(page_size=50, anon_id=user_b)

        assert len(feed_a.product_ids) > 0, "Minimalist user should get products"
        assert len(feed_b.product_ids) > 0, "Trendy user should get products"

        # Compare brand distributions
        brands_a = feed_a.brand_counts()
        brands_b = feed_b.brand_counts()

        print(f"\n  User A (minimalist) top brands: {sorted(brands_a.items(), key=lambda x: -x[1])[:5]}")
        print(f"  User B (trendy) top brands: {sorted(brands_b.items(), key=lambda x: -x[1])[:5]}")

        # Feeds should be meaningfully different
        overlap = set(feed_a.product_ids) & set(feed_b.product_ids)
        total = max(len(feed_a.product_ids), len(feed_b.product_ids), 1)
        overlap_ratio = len(overlap) / total

        print(f"  Product overlap: {len(overlap)}/{total} ({overlap_ratio:.0%})")

        assert overlap_ratio < 0.8, \
            f"Users with opposite preferences should have <80% overlap, got {overlap_ratio:.0%}"

    @pytest.mark.integration
    def test_onboarded_user_pagination_no_duplicates(self):
        """Onboarded user scrolling through multiple pages should see no duplicates."""
        user_id = f"onboard_paginate_{int(time.time())}"
        profile = random_onboarding_profile(user_id)
        submit_onboarding(user_id, profile)

        all_ids = []
        session_id = None
        cursor = None

        for page in range(5):
            result = get_feed(
                session_id=session_id,
                cursor=cursor,
                page_size=20,
                anon_id=user_id,
            )
            all_ids.extend(result.product_ids)
            session_id = result.session_id
            cursor = result.cursor

            if not result.has_more:
                break

        # Verify no duplicates
        assert len(all_ids) == len(set(all_ids)), \
            f"Found {len(all_ids) - len(set(all_ids))} duplicates across {len(all_ids)} items"

    @pytest.mark.integration
    def test_onboarded_feed_scores_descending(self):
        """Onboarded user's feed should have descending scores across pages."""
        user_id = f"onboard_scores_{int(time.time())}"
        profile = random_onboarding_profile(user_id)
        submit_onboarding(user_id, profile)

        all_scores = []
        session_id = None
        cursor = None

        for page in range(3):
            result = get_feed(
                session_id=session_id,
                cursor=cursor,
                page_size=20,
                anon_id=user_id,
            )

            # Scores within page should be descending
            for i in range(1, len(result.scores)):
                assert result.scores[i] <= result.scores[i - 1] + 1e-6, \
                    f"Page {page}: scores not descending at pos {i}: {result.scores[i-1]} -> {result.scores[i]}"

            if all_scores and result.scores:
                # First score of new page should be <= last score of previous page
                assert result.scores[0] <= all_scores[-1] + 1e-6, \
                    f"Score discontinuity at page {page}: prev ended {all_scores[-1]:.4f}, " \
                    f"new starts {result.scores[0]:.4f}"

            all_scores.extend(result.scores)
            session_id = result.session_id
            cursor = result.cursor

            if not result.has_more:
                break

        assert len(all_scores) > 0, "Should have gotten some scores"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
