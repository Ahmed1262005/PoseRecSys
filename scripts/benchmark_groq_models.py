#!/usr/bin/env python3
"""
Benchmark Groq models with the FULL production prompt (15K tokens).

Tests latency and output quality for:
  - meta-llama/llama-4-scout-17b-16e-instruct (current)
  - llama-3.3-70b-versatile
  - llama-3.1-8b-instant
  - qwen/qwen3-32b

Usage:
    PYTHONPATH=src python scripts/benchmark_groq_models.py
"""

import json
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openai import OpenAI
from search.query_planner import _SYSTEM_PROMPT, _get_brand_list_addendum

GROQ_KEY = os.environ["GROQ_API_KEY"]
GROQ_URL = "https://api.groq.com/openai/v1/"

MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
]

QUERIES = [
    "black midi dress for office",
    "something cute for brunch",
    "quiet luxury blazer",
    "floral maxi dress under $50",
]

system_prompt = _SYSTEM_PROMPT + _get_brand_list_addendum()
prompt_chars = len(system_prompt)
prompt_tokens_est = prompt_chars // 4

print(f"System prompt: {prompt_chars:,} chars (~{prompt_tokens_est:,} tokens)")
print(f"Models: {len(MODELS)}")
print(f"Queries: {len(QUERIES)}")
print()


def run_query(client, model, query):
    """Run one planner query, return (latency_ms, valid_json, intent, error)."""
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_object"},
            temperature=0.15,
            max_tokens=1600,
        )
        latency = (time.perf_counter() - t0) * 1000

        raw = resp.choices[0].message.content or ""
        # strip <think> tags for qwen
        if "<think>" in raw:
            think_end = raw.find("</think>")
            if think_end > 0:
                raw = raw[think_end + 8:].strip()

        data = json.loads(raw)
        intent = data.get("intent", "?")
        has_semantic = bool(data.get("semantic_queries") or data.get("semantic_query"))
        has_algolia = "algolia_query" in data
        has_follow_ups = bool(data.get("follow_ups"))
        return latency, True, intent, has_semantic, has_algolia, has_follow_ups, None

    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        return latency, False, "?", False, False, False, str(e)[:80]


def main():
    client = OpenAI(api_key=GROQ_KEY, base_url=GROQ_URL, timeout=30)

    results = {}  # model -> list of (query, latency, valid, intent, ...)

    for model in MODELS:
        print(f"{'='*70}")
        print(f"Model: {model}")
        print(f"{'='*70}")
        model_results = []

        for query in QUERIES:
            latency, valid, intent, has_sem, has_alg, has_fu, err = run_query(client, model, query)
            status = "OK" if valid else f"FAIL: {err}"
            fu_str = "FU" if has_fu else "--"
            print(f"  {latency:>6.0f}ms | {intent:<10} | {fu_str} | {query[:40]:<40} | {status}")
            model_results.append({
                "query": query,
                "latency_ms": latency,
                "valid": valid,
                "intent": intent,
                "has_semantic": has_sem,
                "has_algolia": has_alg,
                "has_follow_ups": has_fu,
                "error": err,
            })

        results[model] = model_results
        avg = sum(r["latency_ms"] for r in model_results) / len(model_results)
        valid_count = sum(1 for r in model_results if r["valid"])
        fu_count = sum(1 for r in model_results if r["has_follow_ups"])
        print(f"  Avg: {avg:.0f}ms | Valid: {valid_count}/{len(model_results)} | Follow-ups: {fu_count}/{len(model_results)}")
        print()

    # Summary table
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Model':<50} {'Avg ms':>8} {'Valid':>6} {'FU':>4}")
    print(f"  {'-'*50} {'-'*8} {'-'*6} {'-'*4}")
    for model in MODELS:
        mrs = results[model]
        avg = sum(r["latency_ms"] for r in mrs) / len(mrs)
        valid = sum(1 for r in mrs if r["valid"])
        fu = sum(1 for r in mrs if r["has_follow_ups"])
        print(f"  {model:<50} {avg:>7.0f}ms {valid:>3}/{len(mrs)} {fu:>2}/{len(mrs)}")


if __name__ == "__main__":
    main()
