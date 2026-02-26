#!/usr/bin/env python3
"""
Diagnostic: trace the profile-aware outfit pipeline step by step.

Shows exactly what happens at each stage:
  1. Source product + target categories
  2. Source-derived prompts vs cluster prompts (text)
  3. Candidate pools from each prompt (who came from where?)
  4. ProfileScorer raw scores per item (why are they all the same?)
  5. Baseline vs personalized final ranking comparison

Usage:
    PYTHONPATH=src python scripts/debug_profile_pipeline.py
"""

import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from supabase import create_client

# ── Setup ──────────────────────────────────────────────────────────
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
SUPABASE = create_client(url, key)

from services.outfit_engine import (
    OutfitEngine,
    AestheticProfile,
    _get_cluster_prompts,
    _profile_to_scoring_dict,
    _gemini_broad,
    compute_compatibility_score,
    get_complementary_targets,
    _filter_by_gemini_category,
    _deduplicate,
    _remove_sets_and_non_outfit,
)
from recs.brand_clusters import (
    BRAND_CLUSTER_MAP,
    CLUSTER_COMPLEMENT_PROMPTS,
    CLUSTER_TRAITS,
    PERSONA_TO_CLUSTERS,
)

ENGINE = OutfitEngine(SUPABASE)

# ── Test config ────────────────────────────────────────────────────

# Use the product from user's test
TEST_PRODUCT_ID = "f3bf95d9-6f9"  # Will be auto-resolved below
TEST_BRANDS = ["Alo Yoga", "Ann Taylor", "Aritzia", "Everlane"]
TEST_PERSONAS = []

# Get a real product with attrs
def find_test_product():
    """Get a random in-stock product with Gemini attrs."""
    result = SUPABASE.table("product_attributes").select(
        "sku_id, category_l1"
    ).eq("category_l1", "Dresses").limit(50).execute()
    if result.data:
        import random
        row = random.choice(result.data)
        return str(row["sku_id"])
    return None


def build_test_profile():
    return {
        "preferred_brands": TEST_BRANDS,
        "brand_openness": "open",
        "style_persona": TEST_PERSONAS,
        "preferred_fits": [], "fit_category_mapping": [],
        "preferred_sleeves": [], "sleeve_category_mapping": [],
        "preferred_lengths": [], "length_category_mapping": [],
        "preferred_lengths_dresses": [],
        "preferred_necklines": [], "preferred_rises": [],
        "top_types": [], "bottom_types": [], "dress_types": [], "outerwear_types": [],
        "patterns_liked": [], "patterns_avoided": [],
        "occasions": [], "colors_to_avoid": [], "styles_to_avoid": [],
        "no_crop": False, "no_revealing": False, "no_sleeveless": False,
        "no_deep_necklines": False, "no_tanks": False,
        "global_min_price": None, "global_max_price": None,
        "birthdate": None, "taste_vector": None,
    }


def run_diagnostic(product_id: str):
    print("=" * 80)
    print(f"DIAGNOSTIC: Profile-Aware Outfit Pipeline")
    print("=" * 80)

    # 1. Fetch source
    print("\n── 1. SOURCE PRODUCT ──")
    source = ENGINE._fetch_product_with_attrs(product_id)
    if not source:
        print(f"  ERROR: Product {product_id} not found")
        return
    print(f"  Name:     {source.name}")
    print(f"  Brand:    {source.brand}")
    print(f"  Category: {source.gemini_category_l1} / {source.gemini_category_l2}")
    print(f"  Style:    {source.style_tags}")
    print(f"  Price:    ${source.price:.0f}")

    source_broad = _gemini_broad(source.gemini_category_l1)
    source.broad_category = source_broad
    all_targets, status = get_complementary_targets(source_broad, source)
    print(f"  Broad:    {source_broad}")
    print(f"  Targets:  {all_targets} (status: {status})")

    # 2. Profile + clusters
    print("\n── 2. USER PROFILE ──")
    profile = build_test_profile()
    clusters = ENGINE._resolve_user_clusters(profile)
    print(f"  Brands:   {TEST_BRANDS}")
    print(f"  Clusters: {clusters}")
    for cid in clusters:
        t = CLUSTER_TRAITS.get(cid)
        if t:
            print(f"    {cid}: {t.name} — {t.style_tags}")

    # 3. For each target category, compare prompts and candidates
    for target_broad in all_targets[:1]:  # Just first target for detail
        print(f"\n── 3. TARGET: {target_broad.upper()} ──")

        # Source-derived prompts
        source_prompts = ENGINE._generate_complement_prompts(source, source_broad, target_broad)
        print(f"\n  Source-derived prompts ({len(source_prompts)}):")
        for i, p in enumerate(source_prompts):
            print(f"    [{i}] {p}")

        # Cluster prompts
        cluster_prompts = _get_cluster_prompts(clusters, target_broad)
        print(f"\n  Cluster prompts ({len(cluster_prompts)}):")
        for i, p in enumerate(cluster_prompts):
            cid = clusters[i] if i < len(clusters) else "?"
            print(f"    [{cid}] {p}")

        # 4. Encode ALL prompts and retrieve per-prompt
        all_prompts = source_prompts + cluster_prompts
        print(f"\n  Total prompts: {len(all_prompts)}")

        # Encode
        vec_strs = ENGINE._encode_texts_batch(all_prompts)
        embeddings_list = [
            [float(x) for x in vs.strip("[]").split(",")]
            for vs in vec_strs
        ]

        # 5. Retrieve with batch RPC — but also retrieve per-prompt to see what each pulls
        print(f"\n── 4. PER-PROMPT RETRIEVAL ──")

        per_prompt_items = {}
        for i, (prompt, emb) in enumerate(zip(all_prompts, embeddings_list)):
            try:
                result = ENGINE._supabase_retry(
                    lambda e=emb: SUPABASE.rpc("batch_complement_search", {
                        "source_product_id": source.product_id,
                        "prompt_embeddings_json": [e],
                        "match_per_prompt": 12,
                        "filter_category": target_broad,
                    }).execute()
                )
                rows = result.data or []
                label = f"source[{i}]" if i < len(source_prompts) else f"cluster[{clusters[i - len(source_prompts)]}]"
                per_prompt_items[label] = rows
                names = [r.get("name", "?")[:40] for r in rows[:5]]
                brands = [r.get("brand", "?") for r in rows[:5]]
                print(f"\n  {label}: \"{prompt[:60]}...\"")
                print(f"    {len(rows)} results. Top 5:")
                for j, (n, b) in enumerate(zip(names, brands)):
                    print(f"      {j+1}. {b} — {n}")
            except Exception as e:
                print(f"  {prompt[:40]}... -> ERROR: {e}")

        # 6. Find unique items per prompt source
        print(f"\n── 5. UNIQUE ITEMS ANALYSIS ──")
        source_ids = set()
        cluster_ids_set = set()
        for label, rows in per_prompt_items.items():
            pids = {str(r.get("product_id", "")) for r in rows}
            if label.startswith("source"):
                source_ids.update(pids)
            else:
                cluster_ids_set.update(pids)

        only_from_clusters = cluster_ids_set - source_ids
        overlap = cluster_ids_set & source_ids
        print(f"  Source prompt pool:  {len(source_ids)} unique items")
        print(f"  Cluster prompt pool: {len(cluster_ids_set)} unique items")
        print(f"  Overlap:             {len(overlap)} items appear in both")
        print(f"  CLUSTER-ONLY items:  {len(only_from_clusters)} NEW items from clusters")

        if only_from_clusters:
            # Show what the cluster-only items look like
            print(f"\n  Cluster-only items (top 10):")
            cluster_only_rows = []
            for label, rows in per_prompt_items.items():
                if label.startswith("cluster"):
                    for r in rows:
                        if str(r.get("product_id", "")) in only_from_clusters:
                            cluster_only_rows.append(r)
            seen = set()
            for r in cluster_only_rows[:10]:
                pid = r.get("product_id", "")
                if pid in seen:
                    continue
                seen.add(pid)
                print(f"    {r.get('brand', '?')} — {r.get('name', '?')[:50]} (${r.get('price', 0):.0f})")

        # 7. ProfileScorer analysis
        print(f"\n── 6. PROFILE SCORING ANALYSIS ──")

        # Get the combined pool (as the real pipeline does)
        try:
            result = ENGINE._supabase_retry(
                lambda: SUPABASE.rpc("batch_complement_search", {
                    "source_product_id": source.product_id,
                    "prompt_embeddings_json": embeddings_list,
                    "match_per_prompt": 12,
                    "filter_category": target_broad,
                }).execute()
            )
            all_rows = result.data or []
        except Exception as e:
            print(f"  Batch retrieval failed: {e}")
            return

        # Deduplicate
        seen_pids = set()
        deduped = []
        for row in all_rows:
            pid = str(row.get("product_id", ""))
            if pid and pid not in seen_pids:
                seen_pids.add(pid)
                deduped.append(row)

        profiles = ENGINE._build_profiles_from_batch_rows(deduped)
        profiles = _filter_by_gemini_category(source, profiles, target_broad)
        profiles = _deduplicate(profiles)
        profiles = _remove_sets_and_non_outfit(source, profiles)

        print(f"  Pool after filtering: {len(profiles)} items")

        # Score each with ProfileScorer
        scorer = ENGINE._get_outfit_scorer()
        score_dist = []
        for cand in profiles[:30]:
            cand.broad_category = _gemini_broad(cand.gemini_category_l1) or target_broad
            cand_dict = _profile_to_scoring_dict(cand)
            adj = scorer.score_item(cand_dict, profile)
            score_dist.append((cand, adj))

        # Sort by adjustment
        score_dist.sort(key=lambda x: x[1], reverse=True)

        print(f"\n  Profile adjustments (top 30 items):")
        adj_values = []
        for cand, adj in score_dist:
            adj_values.append(adj)
            marker = " ***" if adj != score_dist[0][1] else ""
            print(f"    {adj:+.4f}  {cand.brand or '?':20s}  ${cand.price:6.0f}  {(cand.name or '?')[:40]}{marker}")

        unique_adjs = set(round(a, 4) for a in adj_values)
        print(f"\n  Unique adjustment values: {len(unique_adjs)} — {sorted(unique_adjs, reverse=True)}")

        if len(unique_adjs) <= 3:
            print("\n  *** PROBLEM: Almost no differentiation! ***")
            print("  Let's debug WHY by scoring one item in detail...")

            if profiles:
                test_cand = profiles[0]
                test_dict = _profile_to_scoring_dict(test_cand)
                print(f"\n  Test item: {test_cand.brand} — {test_cand.name}")
                print(f"  Item dict: {test_dict}")
                print(f"  Profile keys with values:")
                for k, v in profile.items():
                    if v and v != [] and v != False and v is not None:
                        print(f"    {k}: {v}")

                # Call score_item and also check individual dimensions
                raw = scorer.score_item(test_dict, profile)
                print(f"\n  Raw score: {raw}")
                print(f"  Scorer config max_positive: {scorer.config.max_positive}")
                print(f"  Scorer config max_negative: {scorer.config.max_negative}")


if __name__ == "__main__":
    # Try user's product first, else random
    pid = sys.argv[1] if len(sys.argv) > 1 else find_test_product()
    if not pid:
        print("No product found")
        sys.exit(1)
    print(f"Using product: {pid}")
    run_diagnostic(pid)
