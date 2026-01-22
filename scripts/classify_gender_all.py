"""
Classify all Polyvore-U items by gender using FashionCLIP.

Uses existing CLIP embeddings to efficiently classify items without re-encoding images.
Output: gender_mapping.pkl with {item_idx: 'women'|'men'|'unisex'}
"""
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', default='data/polyvore_u/polyvore_u_clip_embeddings.npy',
                       help='Path to pre-computed CLIP embeddings')
    parser.add_argument('--output', default='data/polyvore_u/gender_mapping.pkl',
                       help='Output path for gender mapping')
    parser.add_argument('--batch_size', type=int, default=10000,
                       help='Batch size for processing')
    args = parser.parse_args()

    print("=" * 60)
    print("FashionCLIP Gender Classification")
    print("=" * 60)

    # Load FashionCLIP for text encoding
    from fashion_clip.fashion_clip import FashionCLIP
    fclip = FashionCLIP('fashion-clip')

    # Gender prompts
    gender_prompts = [
        "women's fashion clothing feminine style dress blouse skirt heels",
        "men's fashion clothing masculine style suit shirt pants shoes",
        "unisex basic clothing t-shirt jeans sneakers hoodie",
    ]

    print("\nEncoding gender prompts...")
    gender_embeddings = fclip.encode_text(gender_prompts, batch_size=3)
    gender_embeddings = gender_embeddings / np.linalg.norm(gender_embeddings, axis=1, keepdims=True)
    print(f"  Gender embeddings shape: {gender_embeddings.shape}")

    # Load pre-computed image embeddings
    print(f"\nLoading image embeddings from: {args.embeddings}")
    embeddings = np.load(args.embeddings)
    print(f"  Loaded {len(embeddings)} embeddings, shape: {embeddings.shape}")

    # Normalize embeddings
    print("\nNormalizing embeddings...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings_normalized = embeddings / norms

    # Classify in batches
    print(f"\nClassifying {len(embeddings)} items...")
    genders = ['women', 'men', 'unisex']
    gender_map = {}
    gender_counts = {'women': 0, 'men': 0, 'unisex': 0}

    for start_idx in tqdm(range(0, len(embeddings), args.batch_size)):
        end_idx = min(start_idx + args.batch_size, len(embeddings))
        batch = embeddings_normalized[start_idx:end_idx]

        # Compute similarities: (batch_size, 3)
        similarities = np.dot(batch, gender_embeddings.T)

        # Get argmax for each item
        predictions = np.argmax(similarities, axis=1)

        for i, pred_idx in enumerate(predictions):
            item_idx = start_idx + i
            gender = genders[pred_idx]
            gender_map[item_idx] = gender
            gender_counts[gender] += 1

    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Women:  {gender_counts['women']:,} ({100*gender_counts['women']/len(embeddings):.1f}%)")
    print(f"  Men:    {gender_counts['men']:,} ({100*gender_counts['men']/len(embeddings):.1f}%)")
    print(f"  Unisex: {gender_counts['unisex']:,} ({100*gender_counts['unisex']/len(embeddings):.1f}%)")
    print(f"  Total:  {len(gender_map):,}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as pickle (dict mapping)
    with open(output_path, 'wb') as f:
        pickle.dump(gender_map, f)
    print(f"\nSaved gender mapping to: {output_path}")

    # Also save separate lists for easy filtering
    women_items = [idx for idx, g in gender_map.items() if g == 'women']
    men_items = [idx for idx, g in gender_map.items() if g == 'men']
    unisex_items = [idx for idx, g in gender_map.items() if g == 'unisex']

    # Save numpy arrays
    np.save(output_path.parent / 'women_item_ids.npy', np.array(women_items))
    np.save(output_path.parent / 'men_item_ids.npy', np.array(men_items))
    np.save(output_path.parent / 'unisex_item_ids.npy', np.array(unisex_items))

    print(f"Saved item lists: women_item_ids.npy, men_item_ids.npy, unisex_item_ids.npy")

    print("\nDone!")

if __name__ == '__main__':
    main()
