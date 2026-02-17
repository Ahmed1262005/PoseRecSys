"""
Test 20 semantic queries through the full hybrid search pipeline.
Shows the LLM planner output + actual product results for each query.

Usage:
    PYTHONPATH=src python scripts/test_20_queries.py
"""

import json
import sys
import time
from dotenv import load_dotenv
load_dotenv()

from search.hybrid_search import HybridSearchService
from search.models import HybridSearchRequest, SortBy


QUERIES = [
    # Texture + category + color
    "fitted ribbed turtleneck in burgundy",
    "chunky cable knit cardigan in cream",
    "pleated satin midi skirt in emerald green",
    "quilted puffer vest in black",

    # Fabric + construction + fit
    "sheer lace bodysuit with long sleeves",
    "silk wrap dress with ruching",
    "linen wide leg palazzo pants",
    "velvet blazer with satin lapels",

    # Pattern + category + detail
    "floral maxi dress with smocked bodice",
    "striped knit polo shirt cropped",
    "plaid wool mini skirt with pleats",
    "polka dot chiffon blouse with bow tie neck",

    # Material + fit + color
    "high waisted wide leg linen pants in olive green",
    "oversized cropped denim jacket with distressing",
    "leather mini skirt with front slit",
    "cashmere v-neck sweater in camel",

    # Construction + texture + style
    "ribbed knit tank top with square neckline",
    "corduroy straight leg trousers in brown",
    "mesh long sleeve top with ruffle trim",
    "off shoulder satin midi dress for a wedding",
]

TOP_N = 5  # Show top N results per query


def run_tests():
    print("Initializing hybrid search service...")
    service = HybridSearchService()
    print("Service ready.\n")

    all_results = []
    total_start = time.time()

    for idx, query in enumerate(QUERIES, 1):
        print(f"{'=' * 100}")
        print(f"[{idx:2d}/20] \"{query}\"")
        print(f"{'=' * 100}")

        request = HybridSearchRequest(
            query=query,
            page=1,
            page_size=20,
            sort_by=SortBy.RELEVANCE,
        )

        t_start = time.time()
        try:
            response = service.search(request)
            elapsed_ms = int((time.time() - t_start) * 1000)

            print(f"\n  Intent:     {response.intent}")
            print(f"  Sort:       {response.sort_by}")
            print(f"  Total:      {response.pagination.total_results or len(response.results)} results")
            print(f"  Timing:     {response.timing}")
            print()

            if response.results:
                print(f"  Top {min(TOP_N, len(response.results))} Results:")
                print(f"  {'#':<4} {'Name':<50} {'Brand':<20} {'Price':>8} {'Pattern':<15} {'Category L2':<20}")
                print(f"  {'-'*4} {'-'*50} {'-'*20} {'-'*8} {'-'*15} {'-'*20}")

                for i, product in enumerate(response.results[:TOP_N], 1):
                    name = (product.name[:47] + "...") if len(product.name) > 50 else product.name
                    brand = (product.brand[:17] + "...") if len(product.brand) > 20 else product.brand
                    pattern = (product.pattern or "-")[:15]
                    cat_l2 = (product.category_l2 or "-")[:20]
                    print(f"  {i:<4} {name:<50} {brand:<20} ${product.price:>7.2f} {pattern:<15} {cat_l2:<20}")

                # Show source distribution
                algolia_count = sum(1 for r in response.results if r.algolia_rank)
                semantic_count = sum(1 for r in response.results if r.semantic_rank)
                both_count = sum(1 for r in response.results if r.algolia_rank and r.semantic_rank)
                print(f"\n  Sources: {algolia_count} Algolia, {semantic_count} Semantic, {both_count} in both")
            else:
                print("  NO RESULTS")

            all_results.append({
                "query": query,
                "intent": response.intent,
                "count": response.pagination.total_results or len(response.results),
                "timing": response.timing,
                "top_results": [
                    {
                        "name": p.name,
                        "brand": p.brand,
                        "price": p.price,
                        "pattern": p.pattern,
                        "category_l2": p.category_l2,
                        "algolia_rank": p.algolia_rank,
                        "semantic_rank": p.semantic_rank,
                    }
                    for p in response.results[:TOP_N]
                ],
                "elapsed_ms": elapsed_ms,
                "success": True,
            })

        except Exception as e:
            elapsed_ms = int((time.time() - t_start) * 1000)
            print(f"\n  ERROR: {e}")
            all_results.append({
                "query": query,
                "elapsed_ms": elapsed_ms,
                "success": False,
                "error": str(e),
            })

        print()

    # Summary
    total_elapsed = int(time.time() - total_start)
    succeeded = sum(1 for r in all_results if r["success"])
    with_results = sum(1 for r in all_results if r.get("count", 0) > 0)

    print(f"\n{'=' * 100}")
    print(f"SUMMARY")
    print(f"{'=' * 100}")
    print(f"  Queries:     {len(QUERIES)}")
    print(f"  Succeeded:   {succeeded}/{len(QUERIES)}")
    print(f"  With results: {with_results}/{len(QUERIES)}")
    print(f"  Total time:  {total_elapsed}s")
    print()

    # Per-query summary table
    print(f"  {'#':<4} {'Query':<50} {'Intent':<10} {'Count':>6} {'Time':>8}")
    print(f"  {'-'*4} {'-'*50} {'-'*10} {'-'*6} {'-'*8}")
    for i, r in enumerate(all_results, 1):
        q = (r["query"][:47] + "...") if len(r["query"]) > 50 else r["query"]
        if r["success"]:
            print(f"  {i:<4} {q:<50} {r['intent']:<10} {r['count']:>6} {r['elapsed_ms']:>7}ms")
        else:
            print(f"  {i:<4} {q:<50} {'FAILED':<10} {'---':>6} {r['elapsed_ms']:>7}ms")


if __name__ == "__main__":
    run_tests()
