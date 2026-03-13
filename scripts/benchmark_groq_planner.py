#!/usr/bin/env python3
"""
Benchmark: Groq Llama 4 Scout planner vs OpenAI gpt-4.1-mini.

Measures real API response times for a variety of query types.
Requires GROQ_API_KEY and OPENAI_API_KEY in .env.

Usage:
    PYTHONPATH=src python scripts/benchmark_groq_planner.py
"""

import json
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Test queries — mix of intents
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    # Heuristic bypass candidates (should skip LLM entirely)
    ("boohoo", "exact-brand"),
    ("dresses", "exact-category"),
    ("black dress", "specific-cat+color"),

    # Must go to LLM
    ("black midi dress for office", "specific-complex"),
    ("quiet luxury blazer", "vague-aesthetic"),
    ("summer vibes date night", "vague-mood"),
    ("floral maxi dress under $50", "specific-with-price"),
    ("something trendy for a wedding", "vague-occasion"),
    ("linen pants casual", "specific-material"),
    ("cottagecore outfit", "vague-aesthetic-2"),
]


def _build_messages(query: str) -> list:
    """Build minimal planner messages (no brand addendum for fairness)."""
    system = (
        "You are a fashion search query planner. Given a user query, return a JSON object with:\n"
        '- "intent": "exact", "specific", or "vague"\n'
        '- "algolia_query": optimized keyword query\n'
        '- "semantic_query": rich visual description for embedding search\n'
        '- "semantic_queries": list of 1-4 diverse semantic queries\n'
        '- "modes": list of style mode tags\n'
        '- "attributes": dict of filter attributes\n'
        '- "avoid": dict of negative filters\n'
        '- "brand": optional exact brand filter\n'
        '- "confidence": 0.0-1.0\n\n'
        "Rules:\n"
        '- If query is JUST a garment type with no styling words, intent MUST be "exact"\n'
        '- If query is JUST a brand name, intent MUST be "exact"\n'
        '- Category + styling attributes = "specific"\n'
        '- Mood/aesthetic only, no garment type = "vague"\n'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]


def benchmark_provider(provider: str, model: str, api_key: str, base_url: str = None):
    """Benchmark a single provider across all test queries."""
    from openai import OpenAI

    client_kwargs = {"api_key": api_key, "timeout": 30.0}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    print(f"\n{'='*70}")
    print(f"  Provider: {provider}  |  Model: {model}")
    print(f"{'='*70}")
    print(f"{'Query':<45} {'Type':<22} {'ms':>8} {'Intent':>8}")
    print(f"{'-'*45} {'-'*22} {'-'*8} {'-'*8}")

    times = []
    errors = 0

    for query, qtype in TEST_QUERIES:
        messages = _build_messages(query)
        t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.15,
                max_tokens=1600,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            content = response.choices[0].message.content
            try:
                parsed = json.loads(content)
                intent = parsed.get("intent", "?")
            except json.JSONDecodeError:
                intent = "BAD_JSON"
            times.append(elapsed_ms)
            print(f"{query:<45} {qtype:<22} {elapsed_ms:>7.0f}ms {intent:>8}")
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            errors += 1
            print(f"{query:<45} {qtype:<22} {elapsed_ms:>7.0f}ms  ERROR: {e}")

    if times:
        print(f"\n  Results: {len(times)} ok, {errors} errors")
        print(f"  Avg:  {sum(times)/len(times):>7.0f}ms")
        print(f"  Min:  {min(times):>7.0f}ms")
        print(f"  Max:  {max(times):>7.0f}ms")
        print(f"  P50:  {sorted(times)[len(times)//2]:>7.0f}ms")
        # Exclude first call (cold start / connection setup)
        if len(times) > 1:
            warm = times[1:]
            print(f"  Avg (warm): {sum(warm)/len(warm):>7.0f}ms")

    return times


def benchmark_heuristic():
    """Benchmark heuristic bypass (no API call)."""
    from search.query_planner import QueryPlanner

    # Create planner with heuristic enabled
    planner = QueryPlanner.__new__(QueryPlanner)
    planner._heuristic_bypass = True

    print(f"\n{'='*70}")
    print(f"  Heuristic Bypass (no LLM call)")
    print(f"{'='*70}")
    print(f"{'Query':<45} {'Type':<22} {'us':>8} {'Hit':>5}")
    print(f"{'-'*45} {'-'*22} {'-'*8} {'-'*5}")

    # Seed brands for heuristic
    import search.query_classifier as qc
    if not qc._BRAND_NAMES:
        try:
            qc.load_brands()
        except Exception:
            # Use fallback brands
            pass

    for query, qtype in TEST_QUERIES:
        iterations = 1000
        t0 = time.perf_counter()
        result = None
        for _ in range(iterations):
            result = planner.try_heuristic_plan(query)
        elapsed_us = ((time.perf_counter() - t0) / iterations) * 1_000_000
        hit = "YES" if result is not None else "no"
        print(f"{query:<45} {qtype:<22} {elapsed_us:>7.0f}us {hit:>5}")


def main():
    groq_key = os.environ.get("GROQ_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    print("Groq Planner Benchmark")
    print(f"Groq key:   {'set' if groq_key else 'MISSING'}")
    print(f"OpenAI key: {'set' if openai_key else 'MISSING'}")

    # 1. Heuristic bypass
    benchmark_heuristic()

    # 2. Groq
    if groq_key:
        benchmark_provider(
            provider="Groq",
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1/",
        )
    else:
        print("\n  Skipping Groq benchmark — GROQ_API_KEY not set")

    # 3. OpenAI (for comparison)
    if openai_key:
        benchmark_provider(
            provider="OpenAI",
            model="gpt-4.1-mini",
            api_key=openai_key,
        )
    else:
        print("\n  Skipping OpenAI benchmark — OPENAI_API_KEY not set")

    # 4. Summary comparison
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print("  Heuristic bypass: ~10-50us (0ms effective)")
    print("  Groq + OpenAI times printed above")
    print("  Target: Groq < 600ms avg, OpenAI ~1500-4000ms avg")


if __name__ == "__main__":
    main()
