"""
Generate FashionCLIP embeddings for Polyvore-U dataset
Maps Fashion-Hash-Net image lists to embeddings indexed by unified ID
"""
import os
import pickle
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path


def load_image_lists(tuples_dir: str):
    """Load image lists from Fashion-Hash-Net format"""
    image_lists = {}
    for cat in ['top', 'bottom', 'shoe']:
        list_path = os.path.join(tuples_dir, f'image_list_{cat}')
        with open(list_path) as f:
            image_lists[cat] = [line.strip() for line in f.readlines()]
        print(f"  {cat}: {len(image_lists[cat])} images")
    return image_lists


def create_unified_mapping(image_lists: dict):
    """
    Create unified item ID mapping from category-specific indices.

    Returns:
        all_paths: list where all_paths[item_id] = image_filename
        category_offset: dict mapping category -> starting item_id
        item_to_category: dict mapping item_id -> (category, local_index)
    """
    all_paths = ['empty_image.png']  # Index 0 is padding/empty
    category_offset = {}
    item_to_category = {}

    current_id = 1
    for cat in ['top', 'bottom', 'shoe']:
        category_offset[cat] = current_id
        for local_idx, img_path in enumerate(image_lists[cat]):
            all_paths.append(img_path)
            item_to_category[current_id] = (cat, local_idx)
            current_id += 1

    print(f"\nUnified mapping created:")
    print(f"  Total items: {len(all_paths) - 1}")  # -1 for padding
    for cat in ['top', 'bottom', 'shoe']:
        print(f"  {cat}: IDs {category_offset[cat]} to {category_offset[cat] + len(image_lists[cat]) - 1}")

    return all_paths, category_offset, item_to_category


def generate_embeddings(
    image_dir: str,
    all_paths: list,
    batch_size: int = 64,
    device: str = 'cuda'
):
    """Generate FashionCLIP embeddings for all images"""
    from fashion_clip.fashion_clip import FashionCLIP

    print("\nLoading FashionCLIP model...")
    fclip = FashionCLIP('fashion-clip')

    # Initialize embeddings array
    n_items = len(all_paths)
    embedding_dim = 512  # FashionCLIP dimension
    embeddings = np.zeros((n_items, embedding_dim), dtype=np.float32)

    # Create empty image embedding for index 0
    empty_img = Image.new('RGB', (224, 224), (255, 255, 255))
    empty_img.save('/tmp/empty_fashion.png')
    empty_emb = fclip.encode_images(['/tmp/empty_fashion.png'], batch_size=1)
    embeddings[0] = empty_emb[0]

    print(f"\nGenerating embeddings for {n_items - 1} images...")

    # Collect valid image paths
    valid_indices = []
    valid_paths = []

    for idx in range(1, n_items):
        img_path = os.path.join(image_dir, all_paths[idx])
        if os.path.exists(img_path):
            valid_indices.append(idx)
            valid_paths.append(img_path)

    print(f"  Found {len(valid_indices)} valid images")

    # Process in batches using FashionCLIP's encode_images
    for batch_start in tqdm(range(0, len(valid_indices), batch_size)):
        batch_end = min(batch_start + batch_size, len(valid_indices))
        batch_indices = valid_indices[batch_start:batch_end]
        batch_paths = valid_paths[batch_start:batch_end]

        try:
            # FashionCLIP handles image loading internally
            batch_embeddings = fclip.encode_images(batch_paths, batch_size=len(batch_paths))

            # Store embeddings
            for i, idx in enumerate(batch_indices):
                embeddings[idx] = batch_embeddings[i]
        except Exception as e:
            print(f"  Error processing batch at {batch_start}: {e}")
            # Fall back to individual processing
            for i, (idx, img_path) in enumerate(zip(batch_indices, batch_paths)):
                try:
                    emb = fclip.encode_images([img_path], batch_size=1)
                    embeddings[idx] = emb[0]
                except Exception as e2:
                    print(f"    Error loading {img_path}: {e2}")
                    embeddings[idx] = embeddings[0]  # Use empty image embedding

    return embeddings


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tuples_dir', default='data/polyvore_u/tuples_630')
    parser.add_argument('--image_dir', default='data/polyvore_u/291x291')
    parser.add_argument('--output_dir', default='data/polyvore_u')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    print("=" * 60)
    print("Polyvore-U Embedding Generator")
    print("=" * 60)

    # Load image lists
    print("\nLoading image lists...")
    image_lists = load_image_lists(args.tuples_dir)

    # Create unified mapping
    all_paths, category_offset, item_to_category = create_unified_mapping(image_lists)

    # Save mapping files
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all_item_image_paths.npy
    paths_file = output_dir / 'all_item_image_paths.npy'
    np.save(paths_file, np.array(all_paths, dtype=object))
    print(f"\nSaved: {paths_file}")

    # Save category offsets
    offsets_file = output_dir / 'category_offsets.pkl'
    with open(offsets_file, 'wb') as f:
        pickle.dump({
            'category_offset': category_offset,
            'item_to_category': item_to_category
        }, f)
    print(f"Saved: {offsets_file}")

    # Generate embeddings
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    embeddings = generate_embeddings(
        args.image_dir,
        all_paths,
        batch_size=args.batch_size,
        device=device
    )

    # Save embeddings
    emb_file = output_dir / 'polyvore_u_clip_embeddings.npy'
    np.save(emb_file, embeddings)
    print(f"\nSaved embeddings: {emb_file}")
    print(f"  Shape: {embeddings.shape}")

    # Also save in pickle format with metadata
    pkl_file = output_dir / 'polyvore_u_embeddings.pkl'
    with open(pkl_file, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'item_ids': list(range(len(all_paths))),
            'image_paths': all_paths,
            'category_offset': category_offset,
        }, f)
    print(f"Saved: {pkl_file}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
