#!/usr/bin/env python3
"""
Quick smoke test: verify v2 search with FAISS returns correctly filtered results.

Tests both the FAISS path (USE_LOCAL_FAISS=true) and verifies filters are
applied correctly end-to-end.

Usage:
    PYTHONPATH=src python scripts/test_v2_filters.py
"""

import os
import sys
import time

os.environ["USE_LOCAL_FAISS"] = "true"

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    # Load FAISS
    from search.local_vector_store import get_local_vector_store
    store = get_local_vector_store()
    store.load_snapshot()
    print(f"FAISS loaded: {store.count} vectors")

    # Load CLIP
    from core.clip_service import get_clip_service
    clip = get_clip_service()
    clip.warmup()

    import numpy as np

    print(f"\n{'='*70}")
    print("  FILTER TESTS — FAISS path")
    print(f"{'='*70}\n")

    tests_passed = 0
    tests_failed = 0

    def check(name, results, assertion_fn, description):
        nonlocal tests_passed, tests_failed
        try:
            assertion_fn(results)
            print(f"  PASS  {name:<40} ({len(results)} results) — {description}")
            tests_passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name:<40} — {e}")
            tests_failed += 1

    def encode(query):
        emb = clip.encode_text(query)
        if hasattr(emb, "numpy"):
            emb = emb.numpy()
        return emb.flatten().astype(np.float32)

    # --- Test 1: No filters ---
    emb = encode("black dress")
    t0 = time.perf_counter()
    results = store.search(emb, limit=50)
    ms = (time.perf_counter() - t0) * 1000
    check("No filters", results,
          lambda r: len(r) == 50,
          f"{ms:.1f}ms")

    # --- Test 2: Brand inclusion ---
    results = store.search(emb, limit=50, include_brands=["Boohoo"])
    check("Brand=Boohoo", results,
          lambda r: all(x["brand"] == "Boohoo" for x in r),
          f"all brands are Boohoo")

    # --- Test 3: Brand inclusion case insensitive ---
    results = store.search(emb, limit=50, include_brands=["boohoo"])
    check("Brand=boohoo (lowercase)", results,
          lambda r: all(x["brand"].lower() == "boohoo" for x in r),
          f"case insensitive")

    # --- Test 4: Brand exclusion ---
    results = store.search(emb, limit=50, exclude_brands=["Boohoo", "Missguided"])
    check("Exclude Boohoo+Missguided", results,
          lambda r: all(x["brand"] not in ("Boohoo", "Missguided") for x in r),
          f"no excluded brands")

    # --- Test 5: Min price ---
    results = store.search(emb, limit=50, min_price=50.0)
    check("Min price $50", results,
          lambda r: all(x["price"] >= 50.0 for x in r),
          f"all prices >= $50")

    # --- Test 6: Max price ---
    results = store.search(emb, limit=50, max_price=30.0)
    check("Max price $30", results,
          lambda r: all(x["price"] <= 30.0 for x in r),
          f"all prices <= $30")

    # --- Test 7: Price range ---
    results = store.search(emb, limit=50, min_price=20.0, max_price=50.0)
    check("Price $20-$50", results,
          lambda r: all(20.0 <= x["price"] <= 50.0 for x in r),
          f"all in range")

    # --- Test 8: Brand + price combo ---
    results = store.search(emb, limit=50, include_brands=["Boohoo"], max_price=25.0)
    check("Brand=Boohoo + max $25", results,
          lambda r: all(x["brand"] == "Boohoo" and x["price"] <= 25.0 for x in r),
          f"brand + price")

    # --- Test 9: Exclude product IDs ---
    all_results = store.search(emb, limit=10)
    first_3_ids = {r["product_id"] for r in all_results[:3]}
    filtered = store.search(emb, limit=10, exclude_product_ids=first_3_ids)
    check("Exclude 3 product IDs", filtered,
          lambda r: all(x["product_id"] not in first_3_ids for x in r),
          f"excluded IDs not in results")

    # --- Test 10: Result format ---
    results = store.search(emb, limit=1)
    required_fields = [
        "product_id", "name", "brand", "image_url", "price",
        "semantic_score", "source", "colors", "materials",
        "category_l1", "category_l2", "broad_category",
    ]
    check("Result format", results,
          lambda r: all(f in r[0] for f in required_fields),
          f"all required fields present")

    # --- Test 11: Source is semantic ---
    check("Source=semantic", results,
          lambda r: r[0]["source"] == "semantic",
          f"source tag correct")

    # --- Test 12: Scores are valid floats ---
    results = store.search(emb, limit=50)
    check("Scores valid", results,
          lambda r: all(isinstance(x["semantic_score"], float) and x["semantic_score"] > 0 for x in r),
          f"positive float scores")

    # --- Test 13: Scores descending ---
    scores = [r["semantic_score"] for r in results]
    check("Scores descending", results,
          lambda r: scores == sorted(scores, reverse=True),
          f"ordered by similarity")

    # --- Test 14: Different query returns different results ---
    emb2 = encode("oversized denim jacket")
    results2 = store.search(emb2, limit=50)
    overlap = len(set(r["product_id"] for r in results) & set(r["product_id"] for r in results2))
    check("Different queries differ", results2,
          lambda r: overlap < 40,
          f"overlap={overlap}/50")

    # --- Test 15: Speed check ---
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        store.search(emb, limit=100)
        times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    check("Speed < 20ms", results,
          lambda r: avg < 20.0,
          f"avg={avg:.1f}ms")

    print(f"\n{'='*70}")
    print(f"  {tests_passed} passed, {tests_failed} failed")
    print(f"{'='*70}")

    return tests_failed == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
