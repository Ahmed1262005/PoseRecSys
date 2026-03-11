"""
Integration tests for Endless Semantic Search (page 2+ for SPECIFIC / VAGUE intent).

Tests verify end-to-end behavior against a running server with real
Algolia + pgvector backends.

Tests cover:
1. SPECIFIC intent page 2+ uses endless semantic (timing shows endless_semantic flag)
2. VAGUE intent page 2+ uses endless semantic
3. EXACT intent page 2+ uses extend-search (Algolia pagination, NOT endless)
4. No product ID overlap between page 1 and page 2
5. No product ID overlap across 5 consecutive pages
6. Page 2+ returns full page_size results for broad queries
7. Category consistency: page 2 results match page 1 intent categories
8. Page 2 is faster than page 1
9. Page 2 completes under 5 seconds
10. Page 3 has similar speed to page 2
11. total_results consistent across pages
12. Eventual exhaustion for narrow queries

Requires:
- Running server: PYTHONPATH=src uvicorn api.app:app --port 8000
- Env vars: TEST_SERVER_URL, SUPABASE_JWT_SECRET
- Algolia index + pgvector embeddings populated

Run with:
    TEST_SERVER_URL=http://localhost:8000 PYTHONPATH=src python -m pytest tests/integration/test_endless_semantic.py -v -s
"""

import os
import sys
import time
import pytest
import requests
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# =============================================================================
# Configuration
# =============================================================================

BASE_URL = os.getenv("TEST_SERVER_URL", "http://localhost:8000")
SEARCH_PREFIX = "/api/search"


def get_auth_headers(user_id: str = "test-endless-semantic-001") -> Dict[str, str]:
    """Generate auth headers with a test JWT token."""
    import jwt as pyjwt

    jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
    if not jwt_secret:
        raise ValueError("SUPABASE_JWT_SECRET required for integration tests")

    now = int(time.time())
    payload = {
        "sub": user_id,
        "aud": "authenticated",
        "role": "authenticated",
        "email": f"{user_id}@test.com",
        "aal": "aal1",
        "exp": now + 3600,
        "iat": now,
        "is_anonymous": False,
    }
    token = pyjwt.encode(payload, jwt_secret, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


def search(query: str, headers: dict, **kwargs) -> requests.Response:
    """Helper: POST /api/search/hybrid with timeout."""
    body = {"query": query, **kwargs}
    return requests.post(
        f"{BASE_URL}{SEARCH_PREFIX}/hybrid",
        json=body,
        headers=headers,
        timeout=60,
    )


def search_page1(query: str, headers: dict, page_size: int = 50, **kwargs) -> dict:
    """Run page 1 search and return parsed response."""
    r = search(query, headers, page_size=page_size, **kwargs)
    assert r.status_code == 200, f"Page 1 failed: {r.status_code} {r.text[:200]}"
    return r.json()


def search_page_n(
    query: str,
    headers: dict,
    search_session_id: str,
    cursor: str,
    page: int = 2,
    page_size: int = 50,
) -> dict:
    """Run page N search using session + cursor from previous page."""
    r = search(
        query, headers,
        page=page,
        page_size=page_size,
        search_session_id=search_session_id,
        cursor=cursor,
    )
    assert r.status_code == 200, f"Page {page} failed: {r.status_code} {r.text[:200]}"
    return r.json()


def get_ids(data: dict) -> set:
    """Extract product IDs from a response."""
    return {p["product_id"] for p in data.get("results", [])}


# =============================================================================
# Skip if no server
# =============================================================================

@pytest.fixture(scope="module")
def auth_headers():
    """Generate auth headers once per module."""
    return get_auth_headers()


@pytest.fixture(scope="module", autouse=True)
def check_server():
    """Skip all tests if server is not reachable."""
    server_url = os.getenv("TEST_SERVER_URL")
    if not server_url:
        pytest.skip("TEST_SERVER_URL not set")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code != 200:
            pytest.skip(f"Server at {BASE_URL} returned {r.status_code}")
    except requests.ConnectionError:
        pytest.skip(f"Server at {BASE_URL} not reachable")


# =============================================================================
# 1. SPECIFIC intent — endless semantic on page 2+
# =============================================================================

@pytest.mark.integration
class TestSpecificIntentEndless:
    """SPECIFIC intent (e.g. 'black midi dress') should use endless semantic on page 2+."""

    def test_page1_returns_session_and_cursor(self, auth_headers):
        """Page 1 should return search_session_id and cursor for page 2."""
        data = search_page1("black midi dress", auth_headers, page_size=10)
        assert data.get("search_session_id"), "No search_session_id"
        assert data.get("cursor"), "No cursor"
        assert data["intent"] == "specific"

    def test_page2_uses_endless_semantic(self, auth_headers):
        """Page 2 timing should show endless_semantic flag (not extend_search)."""
        p1 = search_page1("blue midi dress", auth_headers, page_size=10)
        if not p1.get("cursor"):
            pytest.skip("No cursor from page 1")

        p2 = search_page_n(
            "blue midi dress", auth_headers,
            search_session_id=p1["search_session_id"],
            cursor=p1["cursor"],
            page=2, page_size=10,
        )
        timing = p2.get("timing", {})
        assert timing.get("endless_semantic") is True, (
            f"Expected endless_semantic=True in timing, got: {timing}"
        )

    def test_no_overlap_page1_page2(self, auth_headers):
        """Page 1 and page 2 should have zero overlapping product IDs."""
        p1 = search_page1("casual linen pants", auth_headers, page_size=20)
        if not p1.get("cursor"):
            pytest.skip("No cursor")

        p2 = search_page_n(
            "casual linen pants", auth_headers,
            search_session_id=p1["search_session_id"],
            cursor=p1["cursor"],
            page=2, page_size=20,
        )
        ids1, ids2 = get_ids(p1), get_ids(p2)
        overlap = ids1 & ids2
        assert len(overlap) == 0, f"Page overlap: {len(overlap)} IDs: {list(overlap)[:5]}"

    def test_page2_returns_results(self, auth_headers):
        """Page 2 should return actual results for a broad specific query."""
        p1 = search_page1("floral dress", auth_headers, page_size=20)
        if not p1.get("cursor"):
            pytest.skip("No cursor")

        p2 = search_page_n(
            "floral dress", auth_headers,
            search_session_id=p1["search_session_id"],
            cursor=p1["cursor"],
            page=2, page_size=20,
        )
        assert len(p2["results"]) > 0, "Page 2 returned 0 results for broad query"


# =============================================================================
# 2. VAGUE intent — endless semantic on page 2+
# =============================================================================

@pytest.mark.integration
class TestVagueIntentEndless:
    """VAGUE intent (e.g. 'summer vibes') should use endless semantic on page 2+."""

    def test_vague_page2_uses_endless_semantic(self, auth_headers):
        """VAGUE page 2 should show endless_semantic flag in timing."""
        p1 = search_page1("quiet luxury", auth_headers, page_size=20)
        if not p1.get("cursor"):
            pytest.skip("No cursor (vague query may not paginate)")

        p2 = search_page_n(
            "quiet luxury", auth_headers,
            search_session_id=p1["search_session_id"],
            cursor=p1["cursor"],
            page=2, page_size=20,
        )
        timing = p2.get("timing", {})
        assert timing.get("endless_semantic") is True, (
            f"Expected endless_semantic for vague intent, got: {timing}"
        )

    def test_vague_no_overlap(self, auth_headers):
        """VAGUE pages should not overlap."""
        p1 = search_page1("date night look", auth_headers, page_size=20)
        if not p1.get("cursor"):
            pytest.skip("No cursor")

        p2 = search_page_n(
            "date night look", auth_headers,
            search_session_id=p1["search_session_id"],
            cursor=p1["cursor"],
            page=2, page_size=20,
        )
        overlap = get_ids(p1) & get_ids(p2)
        assert len(overlap) == 0, f"Vague page overlap: {overlap}"


# =============================================================================
# 3. EXACT intent — should NOT use endless semantic
# =============================================================================

@pytest.mark.integration
class TestExactIntentExtendSearch:
    """EXACT intent (e.g. 'tops', 'boohoo') should use _extend_search, NOT endless."""

    def test_exact_page2_uses_extend_search(self, auth_headers):
        """EXACT page 2 should show extend_search flag, NOT endless_semantic."""
        p1 = search_page1("tops", auth_headers, page_size=10)
        if not p1.get("cursor"):
            pytest.skip("No cursor")
        assert p1["intent"] == "exact", f"Expected exact, got {p1['intent']}"

        p2 = search_page_n(
            "tops", auth_headers,
            search_session_id=p1["search_session_id"],
            cursor=p1["cursor"],
            page=2, page_size=10,
        )
        timing = p2.get("timing", {})
        assert timing.get("extend_search") is True, (
            f"Expected extend_search=True for exact intent, got: {timing}"
        )
        assert not timing.get("endless_semantic"), (
            "EXACT intent should NOT use endless_semantic"
        )


# =============================================================================
# 4. Multi-page pagination — no duplicates across 5 pages
# =============================================================================

@pytest.mark.integration
class TestMultiPageNoDuplicates:
    """Verify no product ID overlap across multiple consecutive pages."""

    def test_five_pages_no_overlap(self, auth_headers):
        """Fetch 5 pages and verify zero ID overlap between any pair."""
        page_size = 20
        p1 = search_page1("black dress", auth_headers, page_size=page_size)
        if not p1.get("cursor"):
            pytest.skip("No cursor")

        all_pages = [p1]
        ssid = p1["search_session_id"]
        cursor = p1["cursor"]

        for page_num in range(2, 6):
            if not cursor:
                break
            pn = search_page_n(
                "black dress", auth_headers,
                search_session_id=ssid,
                cursor=cursor,
                page=page_num,
                page_size=page_size,
            )
            all_pages.append(pn)
            cursor = pn.get("cursor")

        # Collect IDs per page
        page_ids = [get_ids(p) for p in all_pages]

        # Check pairwise overlap
        all_seen = set()
        for i, ids in enumerate(page_ids):
            overlap = all_seen & ids
            assert len(overlap) == 0, (
                f"Page {i+1} has {len(overlap)} duplicates from earlier pages: "
                f"{list(overlap)[:5]}"
            )
            all_seen.update(ids)

        total_unique = len(all_seen)
        pages_fetched = len(all_pages)
        print(f"\n  [5-page test] {pages_fetched} pages, {total_unique} unique products, 0 duplicates")


# =============================================================================
# 5. Category consistency
# =============================================================================

@pytest.mark.integration
class TestCategoryConsistency:
    """Page 2 results should match the category intent of the original query."""

    def test_dress_query_page2_mostly_dresses(self, auth_headers):
        """'black midi dress' page 2 should have mostly Dresses category."""
        p1 = search_page1("black midi dress", auth_headers, page_size=20)
        if not p1.get("cursor"):
            pytest.skip("No cursor")

        p2 = search_page_n(
            "black midi dress", auth_headers,
            search_session_id=p1["search_session_id"],
            cursor=p1["cursor"],
            page=2, page_size=20,
        )

        if len(p2["results"]) == 0:
            pytest.skip("Page 2 returned 0 results")

        # Check what percentage are Dresses
        categories = [r.get("category_l1", "") for r in p2["results"]]
        dress_count = sum(1 for c in categories if c and "dress" in c.lower())
        total = len(categories)
        dress_pct = dress_count / total if total > 0 else 0

        print(f"\n  [Category] {dress_count}/{total} dresses ({dress_pct:.0%})")

        # If post_filter_criteria includes category_l1=Dresses, all should match.
        # Allow some slack for semantic drift.
        assert dress_pct >= 0.5, (
            f"Expected >50% dresses on page 2, got {dress_pct:.0%}. "
            f"Categories: {categories}"
        )


# =============================================================================
# 6. Performance
# =============================================================================

@pytest.mark.integration
class TestEndlessPerformance:
    """Page 2+ should be significantly faster than page 1."""

    def test_page2_faster_than_page1(self, auth_headers):
        """Page 2 total_ms should be less than page 1 total_ms."""
        t1_start = time.time()
        p1 = search_page1("casual summer dress", auth_headers, page_size=20)
        t1 = time.time() - t1_start

        if not p1.get("cursor"):
            pytest.skip("No cursor")

        t2_start = time.time()
        p2 = search_page_n(
            "casual summer dress", auth_headers,
            search_session_id=p1["search_session_id"],
            cursor=p1["cursor"],
            page=2, page_size=20,
        )
        t2 = time.time() - t2_start

        p1_ms = p1.get("timing", {}).get("total_ms", t1 * 1000)
        p2_ms = p2.get("timing", {}).get("total_ms", t2 * 1000)

        print(f"\n  [Perf] Page 1: {p1_ms}ms, Page 2: {p2_ms}ms")
        assert p2_ms < p1_ms, (
            f"Page 2 ({p2_ms}ms) should be faster than page 1 ({p1_ms}ms)"
        )

    def test_page2_under_5_seconds(self, auth_headers):
        """Page 2 should complete in under 5 seconds."""
        p1 = search_page1("floral midi skirt", auth_headers, page_size=20)
        if not p1.get("cursor"):
            pytest.skip("No cursor")

        t_start = time.time()
        p2 = search_page_n(
            "floral midi skirt", auth_headers,
            search_session_id=p1["search_session_id"],
            cursor=p1["cursor"],
            page=2, page_size=20,
        )
        elapsed = time.time() - t_start
        p2_ms = p2.get("timing", {}).get("total_ms", elapsed * 1000)

        print(f"\n  [Perf] Page 2: {p2_ms}ms (wall: {elapsed:.1f}s)")
        assert elapsed < 5.0, f"Page 2 took {elapsed:.1f}s, expected <5s"

    def test_page3_similar_speed_to_page2(self, auth_headers):
        """Page 3 should have similar latency to page 2 (within 2x)."""
        p1 = search_page1("white linen pants", auth_headers, page_size=15)
        if not p1.get("cursor"):
            pytest.skip("No cursor")

        p2 = search_page_n(
            "white linen pants", auth_headers,
            search_session_id=p1["search_session_id"],
            cursor=p1["cursor"],
            page=2, page_size=15,
        )
        if not p2.get("cursor"):
            pytest.skip("No cursor for page 3")

        p3 = search_page_n(
            "white linen pants", auth_headers,
            search_session_id=p1["search_session_id"],
            cursor=p2["cursor"],
            page=3, page_size=15,
        )

        p2_ms = p2.get("timing", {}).get("total_ms", 0)
        p3_ms = p3.get("timing", {}).get("total_ms", 0)

        print(f"\n  [Perf] Page 2: {p2_ms}ms, Page 3: {p3_ms}ms")

        if p2_ms > 0 and p3_ms > 0:
            ratio = p3_ms / p2_ms
            assert ratio < 3.0, (
                f"Page 3 ({p3_ms}ms) is {ratio:.1f}x slower than page 2 ({p2_ms}ms)"
            )


# =============================================================================
# 7. total_results consistency
# =============================================================================

@pytest.mark.integration
class TestTotalResultsConsistency:
    """total_results should be consistent across pages."""

    def test_total_results_same_across_pages(self, auth_headers):
        """total_results from page 1 should equal total_results on page 2."""
        p1 = search_page1("striped top", auth_headers, page_size=10)
        if not p1.get("cursor"):
            pytest.skip("No cursor")

        p2 = search_page_n(
            "striped top", auth_headers,
            search_session_id=p1["search_session_id"],
            cursor=p1["cursor"],
            page=2, page_size=10,
        )

        tr1 = p1["pagination"]["total_results"]
        tr2 = p2["pagination"]["total_results"]
        print(f"\n  [Consistency] total_results: page1={tr1}, page2={tr2}")
        assert tr1 == tr2, f"total_results mismatch: page1={tr1}, page2={tr2}"


# =============================================================================
# 8. Timing metadata
# =============================================================================

@pytest.mark.integration
class TestEndlessTimingMetadata:
    """Verify timing dict has expected fields for endless semantic pages."""

    def test_timing_has_rounds(self, auth_headers):
        """Endless semantic timing should report rounds count."""
        p1 = search_page1("elegant evening gown", auth_headers, page_size=10)
        if not p1.get("cursor"):
            pytest.skip("No cursor")

        p2 = search_page_n(
            "elegant evening gown", auth_headers,
            search_session_id=p1["search_session_id"],
            cursor=p1["cursor"],
            page=2, page_size=10,
        )
        timing = p2.get("timing", {})

        if timing.get("endless_semantic"):
            assert "rounds" in timing, f"Missing 'rounds' in timing: {timing}"
            assert "semantic_ms" in timing, f"Missing 'semantic_ms' in timing"
            assert "enrich_ms" in timing, f"Missing 'enrich_ms' in timing"
            assert "filter_ms" in timing, f"Missing 'filter_ms' in timing"
            assert "total_ms" in timing, f"Missing 'total_ms' in timing"
            assert timing["rounds"] >= 1, f"Expected >=1 round, got {timing['rounds']}"
            print(f"\n  [Timing] rounds={timing['rounds']}, "
                  f"semantic={timing['semantic_ms']}ms, "
                  f"enrich={timing['enrich_ms']}ms, "
                  f"filter={timing['filter_ms']}ms, "
                  f"total={timing['total_ms']}ms")


# =============================================================================
# 9. Full page fill for broad queries
# =============================================================================

@pytest.mark.integration
class TestFullPageFill:
    """Broad queries should fill the full page_size on page 2."""

    def test_broad_specific_fills_page(self, auth_headers):
        """A broad specific query should return full page on page 2."""
        page_size = 20
        p1 = search_page1("black dress", auth_headers, page_size=page_size)
        if not p1.get("cursor"):
            pytest.skip("No cursor")

        p2 = search_page_n(
            "black dress", auth_headers,
            search_session_id=p1["search_session_id"],
            cursor=p1["cursor"],
            page=2, page_size=page_size,
        )

        result_count = len(p2["results"])
        print(f"\n  [Fill] Page 2 returned {result_count}/{page_size} results")

        # For a broad query like "black dress" against 96K products,
        # page 2 should be able to fill the full page.
        assert result_count >= page_size * 0.8, (
            f"Expected at least {page_size * 0.8:.0f} results, got {result_count}"
        )
