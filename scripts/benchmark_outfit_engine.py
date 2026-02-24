#!/usr/bin/env python3
"""
Benchmark script for the Complete the Fit outfit engine.

Measures response times across different source categories, cache states,
and reports p50/p95/mean with GPU/CPU detection.

Usage:
    PYTHONPATH=src python scripts/benchmark_outfit_engine.py
    PYTHONPATH=src python scripts/benchmark_outfit_engine.py --rounds 5
"""

import argparse
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def pick_products(supabase, n_per_cat: int = 3):
    """Pick random products with Gemini attrs, grouped by L1 category."""
    cats = ["Tops", "Bottoms", "Dresses", "Outerwear"]
    picks = {}
    for cat in cats:
        rows = (
            supabase.table("product_attributes")
            .select("sku_id")
            .eq("category_l1", cat)
            .limit(50)
            .execute()
            .data or []
        )
        if not rows:
            continue
        random.shuffle(rows)
        # Verify in-stock with image
        verified = []
        for r in rows:
            if len(verified) >= n_per_cat:
                break
            prod = (
                supabase.table("products")
                .select("id, name, brand, in_stock, primary_image_url")
                .eq("id", r["sku_id"])
                .eq("in_stock", True)
                .not_.is_("primary_image_url", "null")
                .limit(1)
                .execute()
            )
            if prod.data:
                verified.append(prod.data[0])
        picks[cat] = verified
    return picks


def run_benchmark(engine, product_id: str):
    """Run build_outfit and return (elapsed_seconds, result_summary)."""
    t0 = time.perf_counter()
    result = engine.build_outfit(product_id=product_id, items_per_category=6)
    elapsed = time.perf_counter() - t0
    n_cats = len(result.get("recommendations", {}))
    n_items = sum(
        len(c.get("items", []))
        for c in result.get("recommendations", {}).values()
    )
    status = result.get("status", "?")
    error = result.get("error")
    return elapsed, {
        "status": status,
        "categories": n_cats,
        "items": n_items,
        "error": error,
    }


def percentile(data, p):
    """Simple percentile calculation."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def main():
    parser = argparse.ArgumentParser(description="Benchmark outfit engine")
    parser.add_argument("--rounds", type=int, default=3, help="Rounds per product (default: 3)")
    parser.add_argument("--products", type=int, default=2, help="Products per category (default: 2)")
    args = parser.parse_args()

    import torch
    from services.outfit_engine import get_outfit_engine

    # ── Environment info ──
    print("=" * 65)
    print("  OUTFIT ENGINE BENCHMARK")
    print("=" * 65)
    print(f"  Device:    {'CUDA — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  GPU Mem:   {mem:.1f} GB")
    print(f"  Rounds:    {args.rounds} per product")
    print(f"  Products:  {args.products} per category")
    print(f"  PyTorch:   {torch.__version__}")
    print("=" * 65)

    # ── Init engine ──
    print("\nInitializing engine...")
    t0 = time.perf_counter()
    engine = get_outfit_engine()
    init_time = time.perf_counter() - t0
    print(f"  Engine init: {init_time:.2f}s")

    # Force model load
    print("Loading FashionCLIP model...")
    t0 = time.perf_counter()
    engine._load_clip()
    clip_time = time.perf_counter() - t0
    print(f"  Model load:  {clip_time:.2f}s")

    # ── Pick products ──
    print(f"\nPicking {args.products} products per category...")
    picks = pick_products(engine.supabase, n_per_cat=args.products)
    for cat, prods in picks.items():
        names = [f"{p['name'][:35]}..." for p in prods]
        print(f"  {cat}: {len(prods)} products — {', '.join(names)}")

    # ── Benchmark ──
    all_times_by_cat = {}
    all_times_global = []

    for cat, products in picks.items():
        cat_times = []
        print(f"\n{'─' * 65}")
        print(f"  {cat.upper()} ({len(products)} products x {args.rounds} rounds)")
        print(f"{'─' * 65}")

        for p_idx, product in enumerate(products):
            pid = product["id"]
            name = product["name"][:45]
            brand = product.get("brand", "?")

            for r in range(args.rounds):
                # Clear cache between rounds to measure consistent perf
                # (comment out next line to test cache-warm performance)
                if r == 0:
                    pass  # first round: cold cache for this product
                # else: cache is warm for repeated prompts

                elapsed, summary = run_benchmark(engine, pid)
                cat_times.append(elapsed)
                all_times_global.append(elapsed)

                tag = "COLD" if r == 0 else "WARM"
                err = f" ERROR={summary['error']}" if summary.get("error") else ""
                print(
                    f"  [{tag}] {name} ({brand}) "
                    f"| {elapsed:5.2f}s "
                    f"| {summary['items']} items / {summary['categories']} cats "
                    f"| {summary['status']}{err}"
                )

        all_times_by_cat[cat] = cat_times

    # ── Summary ──
    print(f"\n{'=' * 65}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 65}")
    print(f"  {'Category':<12} {'Mean':>7} {'P50':>7} {'P95':>7} {'Min':>7} {'Max':>7}  N")
    print(f"  {'─' * 60}")

    for cat, times in all_times_by_cat.items():
        if not times:
            continue
        print(
            f"  {cat:<12} "
            f"{statistics.mean(times):6.2f}s "
            f"{percentile(times, 50):6.2f}s "
            f"{percentile(times, 95):6.2f}s "
            f"{min(times):6.2f}s "
            f"{max(times):6.2f}s "
            f" {len(times)}"
        )

    if all_times_global:
        print(f"  {'─' * 60}")
        print(
            f"  {'ALL':<12} "
            f"{statistics.mean(all_times_global):6.2f}s "
            f"{percentile(all_times_global, 50):6.2f}s "
            f"{percentile(all_times_global, 95):6.2f}s "
            f"{min(all_times_global):6.2f}s "
            f"{max(all_times_global):6.2f}s "
            f" {len(all_times_global)}"
        )

    print(f"\n  Prompt cache entries: {len(engine._embedding_cache)}")
    print(f"  Total benchmark time: {sum(all_times_global):.1f}s")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
