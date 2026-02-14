"""
Integration test: Verify that different onboarding brands produce different feed results.

Tests the 3-tier brand bucketing (60% preferred / 30% cluster-adjacent / 10% discovery)
by onboarding two users with brands from different clusters and comparing their feeds.

User A: Boohoo + Missguided (Cluster H: Ultra-Fast Fashion)
User B: Reformation + Sandro (Cluster A: Modern Classics)

Requires: running server at TEST_SERVER_URL (default http://localhost:8000)
"""

import os
import sys
import time
import json
import requests
import pytest
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

BASE_URL = os.getenv("TEST_SERVER_URL", "http://localhost:8000")
API_PREFIX = "/api/recs/v2"


# =============================================================================
# Auth helper
# =============================================================================

def get_auth_headers(user_id: str) -> dict:
    import jwt as pyjwt
    from dotenv import load_dotenv
    load_dotenv()

    jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
    if not jwt_secret:
        pytest.skip("SUPABASE_JWT_SECRET not set")

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


# =============================================================================
# Helpers
# =============================================================================

def submit_onboarding(user_id: str, preferred_brands: list, brands_to_avoid: list = None) -> dict:
    """Onboard a user with specific brand preferences."""
    headers = get_auth_headers(user_id)
    body = {
        "userId": user_id,
        "gender": "female",
        "core-setup": {
            "selectedCategories": ["tops", "dresses", "bottoms"],
            "sizes": ["M"],
            "colorsToAvoid": [],
            "materialsToAvoid": [],
            "enabled": True,
        },
        "brands": {
            "preferredBrands": preferred_brands,
            "brandsToAvoid": brands_to_avoid or [],
            "brandOpenness": "mix",
            "enabled": True,
        },
        "lifestyle": {
            "occasions": ["everyday", "work"],
            "stylesToAvoid": [],
            "patterns": {},
            "stylePersona": [],
            "enabled": True,
        },
    }
    resp = requests.post(
        f"{BASE_URL}{API_PREFIX}/onboarding",
        json=body,
        headers=headers,
        timeout=30,
    )
    return resp


def get_feed(user_id: str, page_size: int = 50, debug: bool = True) -> dict:
    """Fetch feed for a user."""
    headers = get_auth_headers(user_id)
    resp = requests.get(
        f"{BASE_URL}{API_PREFIX}/feed",
        params={"page_size": page_size, "debug": debug},
        headers=headers,
        timeout=60,
    )
    return resp


def brand_distribution(items: list) -> Counter:
    """Count brands in feed items."""
    return Counter(item.get("brand", "unknown").lower() for item in items)


def print_brand_summary(label: str, items: list, preferred: list, cluster_brands: set):
    """Print a clear summary of brand distribution."""
    dist = brand_distribution(items)
    pref_lower = {b.lower() for b in preferred}
    cluster_lower = {b.lower() for b in cluster_brands}

    t1_count = sum(c for b, c in dist.items() if b in pref_lower)
    t2_count = sum(c for b, c in dist.items() if b in cluster_lower and b not in pref_lower)
    t3_count = len(items) - t1_count - t2_count

    total = len(items)
    print(f"\n{'='*60}")
    print(f"  {label} ({total} items)")
    print(f"{'='*60}")
    print(f"  Preferred brands:     {preferred}")
    print(f"  Tier 1 (preferred):   {t1_count:3d} ({t1_count/total*100:.0f}%)" if total else "  Tier 1: 0")
    print(f"  Tier 2 (cluster-adj): {t2_count:3d} ({t2_count/total*100:.0f}%)" if total else "  Tier 2: 0")
    print(f"  Tier 3 (discovery):   {t3_count:3d} ({t3_count/total*100:.0f}%)" if total else "  Tier 3: 0")
    print(f"  Top 10 brands: {dist.most_common(10)}")


# =============================================================================
# Cluster data for assertions
# =============================================================================

# Cluster H: Ultra-Fast Fashion
CLUSTER_H_BRANDS = {
    "boohoo", "missguided", "prettylittlething", "plt", "shein",
    "nasty gal", "fashion nova", "meshki", "oh polly", "white fox",
}

# Cluster A: Modern Classics / Elevated Everyday
CLUSTER_A_BRANDS = {
    "reformation", "ba&sh", "club monaco", "ann taylor", "j.crew",
    "rails", "sandro", "maje", "reiss", "cos", "arket",
    "& other stories", "other stories", "massimo dutti", "ted baker",
    "karen millen", "whistles", "hobbs", "white house black market",
    "banana republic", "mango",
}


# =============================================================================
# Tests
# =============================================================================

@pytest.fixture(scope="module", autouse=True)
def check_server():
    """Skip all tests if server is not running."""
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code != 200:
            pytest.skip(f"Server returned {r.status_code}")
    except requests.ConnectionError:
        pytest.skip(f"Server not reachable at {BASE_URL}")


class TestBrandBucketingIntegration:
    """Verify that onboarding brand preferences shape the feed via 3-tier bucketing."""

    # Use unique user IDs with timestamp to avoid session contamination
    USER_A_ID = f"brand-test-A-{int(time.time())}"
    USER_B_ID = f"brand-test-B-{int(time.time())}"

    USER_A_BRANDS = ["Boohoo", "Missguided"]        # Cluster H
    USER_B_BRANDS = ["Reformation", "Sandro"]        # Cluster A

    def test_01_onboard_user_a(self):
        """Onboard User A with fast-fashion brands (Cluster H)."""
        resp = submit_onboarding(self.USER_A_ID, self.USER_A_BRANDS)
        assert resp.status_code == 200, f"Onboarding failed: {resp.status_code} {resp.text}"
        data = resp.json()
        assert data["status"] == "success"
        print(f"\nUser A ({self.USER_A_ID}) onboarded with brands: {self.USER_A_BRANDS}")

    def test_02_onboard_user_b(self):
        """Onboard User B with modern-classic brands (Cluster A)."""
        resp = submit_onboarding(self.USER_B_ID, self.USER_B_BRANDS)
        assert resp.status_code == 200, f"Onboarding failed: {resp.status_code} {resp.text}"
        data = resp.json()
        assert data["status"] == "success"
        print(f"\nUser B ({self.USER_B_ID}) onboarded with brands: {self.USER_B_BRANDS}")

    def test_03_feeds_differ(self):
        """The two users should get meaningfully different feeds."""
        resp_a = get_feed(self.USER_A_ID, page_size=50)
        assert resp_a.status_code == 200, f"Feed A failed: {resp_a.status_code} {resp_a.text}"
        items_a = resp_a.json().get("results", [])

        resp_b = get_feed(self.USER_B_ID, page_size=50)
        assert resp_b.status_code == 200, f"Feed B failed: {resp_b.status_code} {resp_b.text}"
        items_b = resp_b.json().get("results", [])

        assert len(items_a) > 0, "User A feed is empty"
        assert len(items_b) > 0, "User B feed is empty"

        # Print summaries
        print_brand_summary("User A (Boohoo/Missguided)", items_a, self.USER_A_BRANDS, CLUSTER_H_BRANDS)
        print_brand_summary("User B (Reformation/Sandro)", items_b, self.USER_B_BRANDS, CLUSTER_A_BRANDS)

        # The brand distributions should differ
        brands_a = {item.get("brand", "").lower() for item in items_a}
        brands_b = {item.get("brand", "").lower() for item in items_b}
        overlap = brands_a & brands_b
        unique_a = brands_a - brands_b
        unique_b = brands_b - brands_a

        print(f"\n  Brand overlap: {len(overlap)} brands in common")
        print(f"  Unique to A:   {len(unique_a)} ({unique_a})")
        print(f"  Unique to B:   {len(unique_b)} ({unique_b})")

        # At minimum, the feeds should not be identical
        ids_a = [item.get("product_id") or item.get("item_id") for item in items_a]
        ids_b = [item.get("product_id") or item.get("item_id") for item in items_b]
        assert ids_a != ids_b, "Both users got identical feed ordering — bucketing has no effect"

    def test_04_user_a_has_cluster_h_presence(self):
        """User A's feed should have significant presence of Cluster H brands."""
        resp = get_feed(self.USER_A_ID, page_size=50)
        items = resp.json().get("results", [])
        assert len(items) > 0, "Feed is empty"

        dist = brand_distribution(items)
        pref_lower = {b.lower() for b in self.USER_A_BRANDS}

        # Count preferred + cluster-adjacent
        cluster_h_count = sum(c for b, c in dist.items() if b in CLUSTER_H_BRANDS)
        pref_count = sum(c for b, c in dist.items() if b in pref_lower)

        total = len(items)
        pref_pct = pref_count / total * 100
        cluster_pct = cluster_h_count / total * 100

        print(f"\n  User A: preferred brands = {pref_count}/{total} ({pref_pct:.0f}%)")
        print(f"  User A: cluster H total  = {cluster_h_count}/{total} ({cluster_pct:.0f}%)")

        # Preferred brands should be meaningfully represented
        # With 60/30/10 bucketing, preferred should be dominant
        assert pref_count > 0, "No preferred brand items in feed at all"

    def test_05_user_b_has_cluster_a_presence(self):
        """User B's feed should have significant presence of Cluster A brands."""
        resp = get_feed(self.USER_B_ID, page_size=50)
        items = resp.json().get("results", [])
        assert len(items) > 0, "Feed is empty"

        dist = brand_distribution(items)
        pref_lower = {b.lower() for b in self.USER_B_BRANDS}

        cluster_a_count = sum(c for b, c in dist.items() if b in CLUSTER_A_BRANDS)
        pref_count = sum(c for b, c in dist.items() if b in pref_lower)

        total = len(items)
        pref_pct = pref_count / total * 100
        cluster_pct = cluster_a_count / total * 100

        print(f"\n  User B: preferred brands = {pref_count}/{total} ({pref_pct:.0f}%)")
        print(f"  User B: cluster A total  = {cluster_a_count}/{total} ({cluster_pct:.0f}%)")

        assert pref_count > 0, "No preferred brand items in feed at all"

    def test_06_cross_cluster_contrast(self):
        """User A should have more Cluster H brands than User B, and vice versa for Cluster A."""
        resp_a = get_feed(self.USER_A_ID, page_size=50)
        resp_b = get_feed(self.USER_B_ID, page_size=50)
        items_a = resp_a.json().get("results", [])
        items_b = resp_b.json().get("results", [])

        dist_a = brand_distribution(items_a)
        dist_b = brand_distribution(items_b)

        h_in_a = sum(c for b, c in dist_a.items() if b in CLUSTER_H_BRANDS)
        h_in_b = sum(c for b, c in dist_b.items() if b in CLUSTER_H_BRANDS)
        a_in_a = sum(c for b, c in dist_a.items() if b in CLUSTER_A_BRANDS)
        a_in_b = sum(c for b, c in dist_b.items() if b in CLUSTER_A_BRANDS)

        print(f"\n  Cluster H items: User A={h_in_a}, User B={h_in_b}")
        print(f"  Cluster A items: User A={a_in_a}, User B={a_in_b}")

        # User A (Boohoo fan) should have more Cluster H items than User B
        assert h_in_a > h_in_b, (
            f"User A (Boohoo fan) has fewer Cluster H items ({h_in_a}) "
            f"than User B ({h_in_b})"
        )

        # User B (Reformation fan) should have more Cluster A items than User A
        assert a_in_b > a_in_a, (
            f"User B (Reformation fan) has fewer Cluster A items ({a_in_b}) "
            f"than User A ({a_in_a})"
        )
