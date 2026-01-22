"""
Generate style exclusion items using FashionCLIP.

For each exclusion option (colors, patterns, item types, shoes), find ~10 representative
items per gender that will be auto-disliked when user selects that option.

Output: data/polyvore_u/style_exclusions.pkl
"""
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse

# All exclusion options with FashionCLIP prompts
EXCLUSION_PROMPTS = {
    'colors': {
        'red': "red colored clothing fashion item",
        'pink': "pink colored clothing fashion item",
        'khaki': "khaki tan beige colored clothing",
        'yellow': "yellow colored clothing fashion item",
        'green': "green colored clothing fashion item",
        'blue': "blue colored clothing fashion item",
        'navy': "navy dark blue colored clothing",
        'purple': "purple violet colored clothing fashion item",
        'brown': "brown colored clothing fashion item",
        'gray': "gray grey colored clothing fashion item",
        'white': "white colored clothing fashion item",
        'black': "black colored clothing fashion item",
    },
    'patterns': {
        'plaids': "plaid tartan checkered pattern clothing",
        'stripes': "striped pattern clothing fashion",
        'dots': "polka dot spotted pattern clothing",
        'floral': "floral flower pattern clothing fashion",
        'geometric': "geometric pattern abstract print clothing",
        'novelty': "novelty print graphic pattern clothing",
    },
    'item_types': {
        'neckwear': "necktie bow tie scarf neckwear accessory",
        'graphic_tshirts': "graphic print t-shirt tee with design",
        'short_sleeves': "short sleeve shirt top",
        'button_down': "button down collar shirt formal",
        'vneck': "v-neck sweater shirt top",
        'pants': "pants trousers slacks",
        'denim': "denim jeans jacket clothing",
        'coats_jackets': "coat jacket outerwear",
        'blazers': "blazer sport coat formal jacket",
        'tank_tops': "tank top sleeveless shirt",
        'belts': "belt leather accessory",
        'shorts': "shorts short pants",
        'socks': "socks hosiery footwear accessory",
        'shoes': "shoes footwear",
    },
    'shoes': {
        'boat_shoes': "boat shoes deck shoes casual footwear",
        'boots': "boots ankle boot footwear",
        'laceup_dress': "lace up dress shoes oxford formal",
        'performance_sneakers': "athletic sneakers running sports shoes",
        'driver': "driver moccasin loafer shoe",
        'loafer': "loafer slip on dress shoe",
        'casual_sneaker': "casual sneakers everyday shoes",
        'sandal': "sandals open toe summer footwear",
    }
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', default='data/polyvore_u/polyvore_u_clip_embeddings.npy',
                       help='Path to pre-computed CLIP embeddings')
    parser.add_argument('--gender_mapping', default='data/polyvore_u/gender_mapping.pkl',
                       help='Path to gender mapping')
    parser.add_argument('--output', default='data/polyvore_u/style_exclusions.pkl',
                       help='Output path for exclusion items')
    parser.add_argument('--items_per_option', type=int, default=10,
                       help='Number of items to select per exclusion option')
    args = parser.parse_args()

    print("=" * 60)
    print("FashionCLIP Style Exclusions Generator")
    print("=" * 60)

    # Load FashionCLIP for text encoding
    from fashion_clip.fashion_clip import FashionCLIP
    fclip = FashionCLIP('fashion-clip')

    # Load pre-computed image embeddings
    print(f"\nLoading image embeddings from: {args.embeddings}")
    embeddings = np.load(args.embeddings)
    print(f"  Loaded {len(embeddings)} embeddings, shape: {embeddings.shape}")

    # Normalize embeddings
    print("Normalizing embeddings...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_normalized = embeddings / norms

    # Load gender mapping
    print(f"\nLoading gender mapping from: {args.gender_mapping}")
    with open(args.gender_mapping, 'rb') as f:
        gender_map = pickle.load(f)

    # Get gender item sets from the mapping format
    women_items = gender_map.get('women_ids', set())
    men_items = gender_map.get('men_ids', set())
    unisex_items = gender_map.get('unisex_ids', set())

    print(f"  Women items: {len(women_items)}")
    print(f"  Men items: {len(men_items)}")
    print(f"  Unisex items: {len(unisex_items)}")

    # For each gender, include unisex items
    women_pool = women_items | unisex_items
    men_pool = men_items | unisex_items

    # Build exclusion data
    exclusions = {
        'colors': {},
        'patterns': {},
        'item_types': {},
        'shoes': {}
    }

    total_options = sum(len(opts) for opts in EXCLUSION_PROMPTS.values())
    print(f"\nProcessing {total_options} exclusion options...")

    for category, options in EXCLUSION_PROMPTS.items():
        print(f"\n--- {category.upper()} ---")
        exclusions[category] = {}

        for option_name, prompt in tqdm(options.items(), desc=category):
            # Encode the prompt
            text_embedding = fclip.encode_text([prompt], batch_size=1)
            text_embedding = text_embedding / np.linalg.norm(text_embedding, axis=1, keepdims=True)

            # Compute similarities with all items
            similarities = np.dot(embeddings_normalized, text_embedding.T).flatten()

            # Get top items for each gender
            exclusions[category][option_name] = {'women': [], 'men': []}

            for gender, pool in [('women', women_pool), ('men', men_pool)]:
                # Get indices sorted by similarity (descending)
                pool_indices = np.array(list(pool))
                pool_similarities = similarities[pool_indices]
                sorted_order = np.argsort(pool_similarities)[::-1]

                # Take top N items
                top_indices = pool_indices[sorted_order[:args.items_per_option]]
                exclusions[category][option_name][gender] = top_indices.tolist()

    # Summary
    print("\n" + "=" * 60)
    print("EXCLUSION ITEMS SUMMARY")
    print("=" * 60)

    for category, options in exclusions.items():
        print(f"\n{category}:")
        for option_name, gender_items in options.items():
            women_count = len(gender_items.get('women', []))
            men_count = len(gender_items.get('men', []))
            print(f"  {option_name}: {women_count} women, {men_count} men")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(exclusions, f)
    print(f"\nSaved exclusion items to: {output_path}")

    # Total items that could be excluded
    total_women = sum(
        len(opts[opt]['women'])
        for opts in exclusions.values()
        for opt in opts
    )
    total_men = sum(
        len(opts[opt]['men'])
        for opts in exclusions.values()
        for opt in opts
    )
    print(f"\nTotal exclusion items: {total_women} women, {total_men} men")
    print("\nDone!")

if __name__ == '__main__':
    main()
