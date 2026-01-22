#!/usr/bin/env python3
"""
Test script for Outrove end-to-end pipeline: Filter -> SASRec Rank
"""
import pickle
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 60)
    print("OUTROVE PIPELINE TEST: Filter -> SASRec Rank")
    print("=" * 60)

    # Step 1: Load tops enriched data
    print("\n[1] Loading tops_enriched.pkl...")
    tops_path = "/home/ubuntu/recSys/outfitTransformer/data/amazon_fashion/processed/tops_enriched.pkl"
    with open(tops_path, 'rb') as f:
        tops_data = pickle.load(f)

    tops_item_ids = set(tops_data['item_ids'])
    print(f"    Loaded {len(tops_item_ids)} tops items")

    # Step 2: Load Amazon Feed to get user history
    print("\n[2] Loading Amazon Fashion Feed (SASRec)...")
    from amazon_feed import AmazonFashionFeed

    amazon_feed = AmazonFashionFeed(
        embeddings_path="/home/ubuntu/recSys/outfitTransformer/data/amazon_fashion/processed/amazon_mens_embeddings.pkl",
        faiss_index_path="/home/ubuntu/recSys/outfitTransformer/models/amazon_mens_faiss.index",
        faiss_ids_path="/home/ubuntu/recSys/outfitTransformer/models/amazon_mens_faiss_ids.npy",
        duorec_checkpoint="/home/ubuntu/recSys/outfitTransformer/models/SASRec-Dec-11-2025_18-20-35.pth",
        user_interactions_path="/home/ubuntu/recSys/outfitTransformer/data/amazon_fashion/recbole/amazon_mens/amazon_mens.inter"
    )

    print(f"    Total users: {len(amazon_feed.user_history)}")
    print(f"    SASRec model loaded: {amazon_feed.seq_model is not None}")

    # Step 3: Find users with tops in their history
    print("\n[3] Finding users with tops in purchase history...")
    users_with_tops = []
    for user_id, items in amazon_feed.user_history.items():
        item_set = set(items)
        tops_in_history = item_set & tops_item_ids
        if len(tops_in_history) >= 3:
            users_with_tops.append({
                'user_id': user_id,
                'tops_count': len(tops_in_history),
                'total_items': len(items),
                'tops_items': list(tops_in_history)[:5]  # Keep first 5 for display
            })

    users_with_tops.sort(key=lambda x: x['tops_count'], reverse=True)
    print(f"    Found {len(users_with_tops)} users with >= 3 tops in history")

    if not users_with_tops:
        print("\n    WARNING: No users found with tops in history!")
        print("    This means SASRec won't be able to rank tops effectively.")
        print("    Falling back to random test...")
        # Use any user
        test_user = list(amazon_feed.user_history.keys())[0]
    else:
        print(f"\n    Top 5 users with most tops:")
        for u in users_with_tops[:5]:
            print(f"      {u['user_id']}: {u['tops_count']} tops / {u['total_items']} total")
        test_user = users_with_tops[0]['user_id']

    # Step 4: Load Outrove filter
    print("\n[4] Loading Outrove Filter...")
    from outrove_filter import OutroveCandidateFilter, OutroveUserProfile

    outrove_filter = OutroveCandidateFilter(tops_path)
    print(f"    Filter loaded with {len(outrove_filter.items)} items")

    # Step 5: Create test profile and get candidates
    print("\n[5] Creating test profile and getting candidates...")
    # Note: from_dict expects flat structure, not nested "global" key
    test_profile_dict = {
        "selectedCoreTypes": ["t-shirts", "hoodies"],
        "colorsToAvoid": ["pink", "yellow"],
        "materialsToAvoid": ["polyester"],
        "tshirts": {
            "fit": "regular",
            "necklines": ["crew", "v-neck"],
            "graphicsTolerance": "minimal",
            "priceRange": [10, 50]
        },
        "hoodies": {
            "priceRange": [20, 80]
        }
    }

    profile = OutroveUserProfile.from_dict(test_profile_dict)
    candidates = outrove_filter.get_candidates(profile)
    print(f"    Filtered candidates: {len(candidates)} items")

    stats = outrove_filter.get_filter_stats(profile)
    print(f"    Filter stats: {stats}")

    # Step 6: Rank with SASRec
    print(f"\n[6] Ranking candidates with SASRec for user '{test_user}'...")
    candidate_list = list(candidates)

    # Get user's history for context
    user_history = amazon_feed.user_history.get(test_user, [])
    print(f"    User history length: {len(user_history)}")

    # Rank candidates
    ranked = amazon_feed.rank_with_sasrec(test_user, candidate_list, topk=20)

    print(f"\n    Top 20 ranked items:")
    print("-" * 60)

    non_zero_scores = 0
    for i, (item_id, score) in enumerate(ranked[:20]):
        # Get item info
        item_info = outrove_filter.items.get(item_id, {})
        category = item_info.get('outrove_type', 'unknown')
        colors = item_info.get('colors', [])
        price = item_info.get('price')

        if score > 0:
            non_zero_scores += 1

        print(f"    {i+1:2}. {item_id[:15]:15} | score: {score:.4f} | cat: {category:10} | ${price or 'N/A'}")

    # Step 7: Summary
    print("\n" + "=" * 60)
    print("PIPELINE TEST SUMMARY")
    print("=" * 60)
    print(f"  Tops items loaded:     {len(tops_item_ids)}")
    print(f"  Users with tops hist:  {len(users_with_tops)}")
    print(f"  Test user:             {test_user}")
    print(f"  Filtered candidates:   {len(candidates)}")
    print(f"  Items with score > 0:  {non_zero_scores} / 20")

    if non_zero_scores == 0:
        print("\n  WARNING: All scores are 0.0!")
        print("  This likely means the user has no overlap with candidate items.")
        print("  SASRec needs user history that overlaps with candidates.")
    elif non_zero_scores < 10:
        print(f"\n  PARTIAL: Only {non_zero_scores} items got non-zero scores.")
    else:
        print(f"\n  SUCCESS: SASRec ranking is working!")

    print("\n" + "=" * 60)
    return ranked

if __name__ == "__main__":
    main()
