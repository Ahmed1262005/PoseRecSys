#!/usr/bin/env python3
"""
Benchmark: Local FAISS vs Supabase pgvector for semantic search.

Loads the FAISS snapshot, encodes test queries with FashionCLIP, and
compares search latency + result overlap between FAISS and pgvector.

Usage:
    PYTHONPATH=src python scripts/benchmark_faiss_vs_pgvector.py
"""

import os
import sys
import time

import numpy as np
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


TEST_QUERIES = [
    "black midi dress",
    "casual linen pants",
    "quiet luxury blazer",
    "summer floral maxi dress",
    "elegant evening gown",
    "oversized denim jacket",
    "red cocktail dress",
    "cozy knit sweater",
    "boohoo bodycon dress",
    "minimalist white blouse",
]

FILTER_TESTS = [
    # (query, kwargs_label, kwargs)
    ("black dress", "no filters", {}),
    ("black dress", "brand=Boohoo", {"include_brands": ["Boohoo"]}),
    ("dress", "price $20-50", {"min_price": 20, "max_price": 50}),
    ("top", "exclude Boohoo+Missguided", {"exclude_brands": ["Boohoo", "Missguided"]}),
    ("jacket", "brand=Zara + price<100", {"include_brands": ["Zara"], "max_price": 100}),
]


def main():
    # 1. Load FAISS snapshot
    print("Loading FAISS snapshot...")
    t0 = time.perf_counter()
    from search.local_vector_store import get_local_vector_store
    store = get_local_vector_store()
    store.load_snapshot()
    faiss_load_ms = (time.perf_counter() - t0) * 1000
    print(f"  FAISS loaded: {store.count} vectors in {faiss_load_ms:.0f}ms")

    # 2. Load FashionCLIP for encoding
    print("\nLoading FashionCLIP...")
    from core.clip_service import get_clip_service
    clip = get_clip_service()
    clip.warmup()
    print("  FashionCLIP ready")

    # 3. Load pgvector engine for comparison
    print("\nLoading pgvector engine...")
    from women_search_engine import get_women_search_engine
    pgvector_engine = get_women_search_engine()
    print("  pgvector engine ready")

    # ==================================================================
    # Benchmark 1: Basic search (no filters)
    # ==================================================================
    print(f"\n{'='*80}")
    print("  BENCHMARK 1: Basic Search (limit=100, no filters)")
    print(f"{'='*80}")
    print(f"{'Query':<35} {'FAISS ms':>10} {'pgvector ms':>12} {'Speedup':>9} {'Overlap':>9}")
    print(f"{'-'*35} {'-'*10} {'-'*12} {'-'*9} {'-'*9}")

    faiss_times = []
    pgvector_times = []

    for query in TEST_QUERIES:
        # Encode query
        embedding = clip.encode_text(query)
        if hasattr(embedding, "numpy"):
            embedding = embedding.numpy()
        embedding = embedding.flatten().astype(np.float32)

        # FAISS search
        t0 = time.perf_counter()
        faiss_results = store.search(embedding, limit=100)
        faiss_ms = (time.perf_counter() - t0) * 1000
        faiss_times.append(faiss_ms)

        # pgvector search
        t0 = time.perf_counter()
        pg_resp = pgvector_engine.search_multimodal(
            query=query,
            limit=100,
            query_embedding=embedding,
        )
        pg_ms = (time.perf_counter() - t0) * 1000
        pgvector_times.append(pg_ms)

        # Compute result overlap
        faiss_ids = {r["product_id"] for r in faiss_results}
        pg_ids = {r["product_id"] for r in pg_resp.get("results", [])}
        overlap = len(faiss_ids & pg_ids) / max(len(faiss_ids | pg_ids), 1) * 100

        speedup = pg_ms / max(faiss_ms, 0.01)
        print(f"{query:<35} {faiss_ms:>9.1f}ms {pg_ms:>11.1f}ms {speedup:>8.0f}x {overlap:>7.0f}%")

    print(f"\n  FAISS  avg: {sum(faiss_times)/len(faiss_times):.1f}ms  |  min: {min(faiss_times):.1f}ms  |  max: {max(faiss_times):.1f}ms")
    print(f"  pgvec  avg: {sum(pgvector_times)/len(pgvector_times):.0f}ms  |  min: {min(pgvector_times):.0f}ms  |  max: {max(pgvector_times):.0f}ms")
    print(f"  Avg speedup: {sum(pgvector_times)/max(sum(faiss_times), 0.01):.0f}x")

    # ==================================================================
    # Benchmark 2: Filtered search
    # ==================================================================
    print(f"\n{'='*80}")
    print("  BENCHMARK 2: Filtered Search (limit=100)")
    print(f"{'='*80}")
    print(f"{'Query + Filter':<45} {'FAISS ms':>10} {'Results':>9}")
    print(f"{'-'*45} {'-'*10} {'-'*9}")

    for query, label, kwargs in FILTER_TESTS:
        embedding = clip.encode_text(query)
        if hasattr(embedding, "numpy"):
            embedding = embedding.numpy()
        embedding = embedding.flatten().astype(np.float32)

        t0 = time.perf_counter()
        results = store.search(embedding, limit=100, **kwargs)
        ms = (time.perf_counter() - t0) * 1000

        desc = f"{query} ({label})"
        print(f"{desc:<45} {ms:>9.1f}ms {len(results):>9}")

    # ==================================================================
    # Benchmark 3: Throughput (simulated multi-query)
    # ==================================================================
    print(f"\n{'='*80}")
    print("  BENCHMARK 3: Multi-query throughput (3 queries, sequential)")
    print(f"{'='*80}")

    queries_3 = ["black midi dress", "casual summer top", "elegant evening outfit"]
    embeddings_3 = []
    for q in queries_3:
        e = clip.encode_text(q)
        if hasattr(e, "numpy"):
            e = e.numpy()
        embeddings_3.append(e.flatten().astype(np.float32))

    # FAISS: 3 sequential searches
    t0 = time.perf_counter()
    for emb in embeddings_3:
        store.search(emb, limit=100)
    faiss_3q_ms = (time.perf_counter() - t0) * 1000

    # pgvector: 3 sequential searches
    t0 = time.perf_counter()
    for q, emb in zip(queries_3, embeddings_3):
        pgvector_engine.search_multimodal(query=q, limit=100, query_embedding=emb)
    pg_3q_ms = (time.perf_counter() - t0) * 1000

    print(f"  FAISS  (3 queries): {faiss_3q_ms:>8.1f}ms")
    print(f"  pgvec  (3 queries): {pg_3q_ms:>8.0f}ms")
    print(f"  Speedup: {pg_3q_ms / max(faiss_3q_ms, 0.01):.0f}x")

    # ==================================================================
    # Benchmark 4: Snapshot load time
    # ==================================================================
    print(f"\n{'='*80}")
    print("  BENCHMARK 4: Snapshot load time (cold start)")
    print(f"{'='*80}")

    # Reset singleton and reload
    from search.local_vector_store import LocalVectorStore
    LocalVectorStore._instance = None

    t0 = time.perf_counter()
    store2 = get_local_vector_store()
    store2.load_snapshot()
    reload_ms = (time.perf_counter() - t0) * 1000
    print(f"  Snapshot reload: {reload_ms:.0f}ms ({store2.count} vectors)")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    print(f"  Vectors indexed:    {store.count:,}")
    print(f"  Snapshot size:      ~334MB on disk")
    print(f"  Snapshot load:      {reload_ms:.0f}ms")
    print(f"  FAISS avg search:   {sum(faiss_times)/len(faiss_times):.1f}ms")
    print(f"  pgvector avg search:{sum(pgvector_times)/len(pgvector_times):.0f}ms")
    print(f"  Avg speedup:        {sum(pgvector_times)/max(sum(faiss_times), 0.01):.0f}x")


if __name__ == "__main__":
    main()
