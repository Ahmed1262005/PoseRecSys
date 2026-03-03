"""
Diagnostic test: trace the full refinement flow through /hybrid.

Simulates: "vacation outfits" -> user selects Dresses + Smart Casual
-> calls /hybrid with selected_filters + selection_labels
-> checks that results are actually Dresses.

Run:
    PYTHONPATH=src python scripts/test_refinement_flow.py
"""

import os
import sys
import json
import requests
import jwt
import time

API_URL = os.getenv("TEST_SERVER_URL", "http://localhost:8000")
SEARCH_URL = f"{API_URL}/api/search"

# Generate a test JWT
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
if not SUPABASE_JWT_SECRET:
    print("ERROR: SUPABASE_JWT_SECRET not set")
    sys.exit(1)

now = int(time.time())
token = jwt.encode(
    {
        "sub": "test-refinement-user",
        "aud": "authenticated",
        "role": "authenticated",
        "email": "test-refinement@test.com",
        "aal": "aal1",
        "exp": now + 86400,
        "iat": now,
        "is_anonymous": False,
    },
    SUPABASE_JWT_SECRET,
    algorithm="HS256",
)
HEADERS = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def sep(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def test_initial_search():
    """Step 1: Initial search for 'vacation outfits' (no refinement)."""
    sep("STEP 1: Initial search — 'vacation outfits'")

    body = {
        "query": "vacation outfits",
        "page": 1,
        "page_size": 10,
    }

    resp = requests.post(f"{SEARCH_URL}/hybrid", json=body, headers=HEADERS, timeout=120)
    data = resp.json()

    timing = data.get("timing", {})
    print(f"Status:  {resp.status_code}")
    print(f"Results: {data.get('total_results', '?')}")
    print(f"Planner: {timing.get('planner_ms', '?')}ms")
    print(f"Intent:  {timing.get('plan_intent', '?')}")
    print(f"Modes:   {timing.get('plan_modes', [])}")
    print(f"Applied: {timing.get('plan_applied_filters', {})}")

    # Show follow-ups
    follow_ups = data.get("follow_ups", [])
    print(f"\nFollow-ups ({len(follow_ups)}):")
    for i, fu in enumerate(follow_ups):
        q = fu.get("question", "?")
        opts = fu.get("options", [])
        opt_labels = [o.get("label", "?") for o in opts]
        print(f"  Q{i}: {q}")
        print(f"      Options: {opt_labels}")

    # Show first 5 results with categories
    results = data.get("results", [])
    print(f"\nFirst 5 results:")
    for r in results[:5]:
        name = r.get("name", "?")[:60]
        cat = r.get("category_l1", "?")
        cat2 = r.get("category_l2", "?")
        score = r.get("score", 0)
        source = r.get("source", "?")
        print(f"  [{source}] {cat}/{cat2} | {score:.3f} | {name}")

    return data


def test_refinement(follow_ups_data=None):
    """Step 2: Refinement — select Dresses + Smart Casual."""
    sep("STEP 2: Refinement — selected_filters={category_l1: [Dresses], formality: [Smart Casual]}")

    body = {
        "query": "vacation outfits",
        "page": 1,
        "page_size": 30,
        "selected_filters": {
            "category_l1": ["Dresses"],
            "formality": ["Smart Casual"],
        },
        "selection_labels": ["Dresses", "Smart Casual"],
    }

    resp = requests.post(f"{SEARCH_URL}/hybrid", json=body, headers=HEADERS, timeout=120)
    data = resp.json()

    if "error" in data:
        print(f"ERROR: {data['error']}")
        return data

    timing = data.get("timing", {})
    print(f"Status:  {resp.status_code}")
    print(f"Results: {data.get('total_results', '?')}")
    print(f"Planner: {timing.get('planner_ms', '?')}ms")
    print(f"Intent:  {timing.get('plan_intent', '?')}")
    print(f"Modes:   {timing.get('plan_modes', [])}")
    print(f"Applied: {timing.get('plan_applied_filters', {})}")
    print(f"Algolia ms: {timing.get('algolia_ms', '?')}")
    print(f"Semantic ms: {timing.get('semantic_ms', '?')}")

    # Check plan details
    plan_attrs = timing.get("plan_attributes", {})
    plan_avoid = timing.get("plan_avoid", {})
    plan_sq = timing.get("plan_semantic_queries", [])
    print(f"\nPlan attributes: {json.dumps(plan_attrs, indent=2)}")
    print(f"Plan avoid: {json.dumps(plan_avoid, indent=2)}")
    print(f"Plan semantic queries: {json.dumps(plan_sq, indent=2)}")

    # Follow-ups
    follow_ups = data.get("follow_ups", [])
    print(f"\nNew follow-ups ({len(follow_ups)}):")
    for i, fu in enumerate(follow_ups):
        q = fu.get("question", "?")
        opts = fu.get("options", [])
        opt_labels = [o.get("label", "?") for o in opts]
        print(f"  Q{i}: {q}")
        print(f"      Options: {opt_labels}")

    # Results with category breakdown
    results = data.get("results", [])
    cat_counts = {}
    print(f"\nAll {len(results)} results:")
    for i, r in enumerate(results):
        name = r.get("name", "?")[:50]
        cat = r.get("category_l1", "MISSING")
        cat2 = r.get("category_l2", "?")
        score = r.get("score", 0)
        source = r.get("source", "?")
        brand = r.get("brand", "?")
        print(f"  {i+1:2d}. [{source:8s}] {cat:12s}/{cat2:20s} | {score:.3f} | {brand:20s} | {name}")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    # Category summary
    print(f"\nCategory breakdown:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100 if results else 0
        is_dress = "OK" if cat == "Dresses" else "WRONG"
        print(f"  {cat:20s}: {count:3d} ({pct:.0f}%) {is_dress}")

    dress_count = cat_counts.get("Dresses", 0)
    total = len(results)
    if total > 0 and dress_count == total:
        print(f"\n  PASS: All {total} results are Dresses")
    elif total > 0:
        non_dress = total - dress_count
        print(f"\n  FAIL: {non_dress}/{total} results are NOT Dresses")
    else:
        print(f"\n  FAIL: No results returned")

    return data


def test_refinement_with_modes():
    """Step 3: Refinement with modes — select Dresses + cover_arms."""
    sep("STEP 3: Refinement with modes — selected_filters={category_l1: [Dresses], modes: [cover_arms]}")

    body = {
        "query": "vacation outfits",
        "page": 1,
        "page_size": 20,
        "selected_filters": {
            "category_l1": ["Dresses"],
            "modes": ["cover_arms"],
        },
        "selection_labels": ["Dresses", "Covered arms"],
    }

    resp = requests.post(f"{SEARCH_URL}/hybrid", json=body, headers=HEADERS, timeout=120)
    data = resp.json()

    if "error" in data:
        print(f"ERROR: {data['error']}")
        return data

    timing = data.get("timing", {})
    print(f"Status:  {resp.status_code}")
    print(f"Results: {data.get('total_results', '?')}")
    print(f"Intent:  {timing.get('plan_intent', '?')}")
    print(f"Modes:   {timing.get('plan_modes', [])}")
    print(f"Applied: {timing.get('plan_applied_filters', {})}")

    results = data.get("results", [])
    cat_counts = {}
    for r in results:
        cat = r.get("category_l1", "MISSING")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    print(f"\nCategory breakdown:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        is_dress = "OK" if cat == "Dresses" else "WRONG"
        print(f"  {cat:20s}: {count:3d} {is_dress}")

    dress_count = cat_counts.get("Dresses", 0)
    total = len(results)
    if total > 0 and dress_count == total:
        print(f"\n  PASS: All {total} results are Dresses")
    else:
        non_dress = total - dress_count
        print(f"\n  FAIL: {non_dress}/{total} results are NOT Dresses")

    return data


if __name__ == "__main__":
    print("Refinement Flow Diagnostic Test")
    print(f"API: {API_URL}")
    print(f"Token: {token[:20]}...")

    # Check server health first
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        print(f"Server health: {r.status_code}")
    except Exception as e:
        print(f"Server not reachable: {e}")
        sys.exit(1)

    data1 = test_initial_search()
    data2 = test_refinement()
    data3 = test_refinement_with_modes()

    sep("SUMMARY")
    print("Step 1 (initial):       OK" if data1.get("results") else "Step 1: FAIL")
    r2 = data2.get("results", [])
    r2_dresses = sum(1 for r in r2 if r.get("category_l1") == "Dresses")
    print(f"Step 2 (Dresses+SC):    {r2_dresses}/{len(r2)} Dresses {'PASS' if r2_dresses == len(r2) and len(r2) > 0 else 'FAIL'}")
    r3 = data3.get("results", [])
    r3_dresses = sum(1 for r in r3 if r.get("category_l1") == "Dresses")
    print(f"Step 3 (Dresses+modes): {r3_dresses}/{len(r3)} Dresses {'PASS' if r3_dresses == len(r3) and len(r3) > 0 else 'FAIL'}")
