"""
Women's Fashion FashionCLIP Embeddings Generator

Generates 512-dimensional FashionCLIP embeddings for all women's fashion items.

Input: data/women_fashion/processed/women_items.pkl
Output: models/women_embeddings.pkl
"""

import os
import pickle
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm

# Paths
BASE_DIR = Path("/home/ubuntu/recSys/outfitTransformer")
ITEMS_PATH = BASE_DIR / "data" / "women_fashion" / "processed" / "women_items.pkl"
EMBEDDINGS_PATH = BASE_DIR / "models" / "women_embeddings.pkl"


def load_fashion_clip():
    """Load FashionCLIP model."""
    print("Loading FashionCLIP model...")
    try:
        from fashion_clip.fashion_clip import FashionCLIP
        model = FashionCLIP('fashion-clip')
        print("  FashionCLIP loaded via fashion_clip library")
        return model, 'fashion_clip'
    except ImportError:
        print("  fashion_clip not available, trying transformers...")
        from transformers import CLIPProcessor, CLIPModel
        import torch
        model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        print("  CLIP model loaded via transformers")
        return (model, processor), 'transformers'


def encode_image_fashion_clip(model, image_path: str) -> Optional[np.ndarray]:
    """Encode a single image using FashionCLIP."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Encode using FashionCLIP
            if hasattr(model, 'encode_images'):
                embedding = model.encode_images([img], batch_size=1)
                embedding = np.array(embedding[0])
            else:
                # Should not happen but fallback
                return None

            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            return embedding

    except Exception as e:
        print(f"  Error encoding {image_path}: {e}")
        return None


def encode_image_transformers(model_tuple, image_path: str) -> Optional[np.ndarray]:
    """Encode a single image using transformers CLIP."""
    import torch

    model, processor = model_tuple

    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                embedding = image_features.cpu().numpy()[0]

            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            return embedding

    except Exception as e:
        print(f"  Error encoding {image_path}: {e}")
        return None


def generate_embeddings(
    items_path: Path = ITEMS_PATH,
    output_path: Path = EMBEDDINGS_PATH,
    batch_size: int = 32
) -> Dict:
    """Generate embeddings for all women's fashion items."""
    print("=" * 60)
    print("Women's Fashion Embeddings Generator")
    print("=" * 60)

    # Load items
    print(f"\nLoading items from {items_path}...")
    with open(items_path, 'rb') as f:
        data = pickle.load(f)

    items = data['items']
    print(f"  Loaded {len(items)} items")

    # Load model
    model, model_type = load_fashion_clip()

    # Choose encoding function
    if model_type == 'fashion_clip':
        encode_fn = lambda img_path: encode_image_fashion_clip(model, img_path)
    else:
        encode_fn = lambda img_path: encode_image_transformers(model, img_path)

    # Generate embeddings
    print(f"\nGenerating embeddings for {len(items)} items...")
    embeddings = {}
    failed = []

    for item_id, item_data in tqdm(items.items(), desc="Encoding"):
        image_path = item_data.get('image_path')
        if not image_path or not Path(image_path).exists():
            failed.append(item_id)
            continue

        embedding = encode_fn(image_path)
        if embedding is not None:
            embeddings[item_id] = {
                'embedding': embedding,
                'category': item_data.get('category', ''),
                'subcategory': item_data.get('subcategory', ''),
                'filename': Path(image_path).name,
                'path': image_path,
            }
        else:
            failed.append(item_id)

    print(f"\n  Successfully encoded: {len(embeddings)}")
    print(f"  Failed: {len(failed)}")

    # Save embeddings
    print(f"\nSaving embeddings to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    # Verify
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved {file_size:.1f} MB")
    print("Done!")

    return embeddings


def batch_encode_fashion_clip(model, image_paths: list, batch_size: int = 32) -> Dict[str, np.ndarray]:
    """Batch encode images using FashionCLIP for better efficiency."""
    from PIL import Image
    import numpy as np

    embeddings = {}
    failed = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        valid_paths = []

        for path in batch_paths:
            try:
                with Image.open(path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    batch_images.append(img.copy())
                    valid_paths.append(path)
            except Exception as e:
                failed.append(path)

        if batch_images:
            try:
                batch_embeddings = model.encode_images(batch_images, batch_size=len(batch_images))
                for path, emb in zip(valid_paths, batch_embeddings):
                    emb = np.array(emb)
                    emb = emb / np.linalg.norm(emb)
                    embeddings[path] = emb
            except Exception as e:
                print(f"  Batch encoding error: {e}")
                failed.extend(valid_paths)

    return embeddings, failed


def generate_embeddings_batched(
    items_path: Path = ITEMS_PATH,
    output_path: Path = EMBEDDINGS_PATH,
    batch_size: int = 32
) -> Dict:
    """Generate embeddings using batched encoding for better performance."""
    print("=" * 60)
    print("Women's Fashion Embeddings Generator (Batched)")
    print("=" * 60)

    # Load items
    print(f"\nLoading items from {items_path}...")
    with open(items_path, 'rb') as f:
        data = pickle.load(f)

    items = data['items']
    print(f"  Loaded {len(items)} items")

    # Load model
    model, model_type = load_fashion_clip()

    if model_type != 'fashion_clip':
        print("  Batched encoding only supported with fashion_clip, falling back to sequential")
        return generate_embeddings(items_path, output_path, batch_size)

    # Collect image paths
    item_paths = {}
    for item_id, item_data in items.items():
        image_path = item_data.get('image_path')
        if image_path and Path(image_path).exists():
            item_paths[item_id] = image_path

    print(f"  Found {len(item_paths)} valid image paths")

    # Batch encode
    print(f"\nGenerating embeddings in batches of {batch_size}...")
    path_list = list(item_paths.values())
    path_embeddings, failed = batch_encode_fashion_clip(model, path_list, batch_size)

    # Map back to item IDs
    embeddings = {}
    for item_id, image_path in tqdm(item_paths.items(), desc="Mapping"):
        if image_path in path_embeddings:
            item_data = items[item_id]
            embeddings[item_id] = {
                'embedding': path_embeddings[image_path],
                'category': item_data.get('category', ''),
                'subcategory': item_data.get('subcategory', ''),
                'filename': Path(image_path).name,
                'path': image_path,
            }

    print(f"\n  Successfully encoded: {len(embeddings)}")
    print(f"  Failed: {len(failed)}")

    # Save embeddings
    print(f"\nSaving embeddings to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    # Verify
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved {file_size:.1f} MB")
    print("Done!")

    return embeddings


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate women\'s fashion embeddings')
    parser.add_argument(
        '--items',
        default=str(ITEMS_PATH),
        help='Path to women_items.pkl'
    )
    parser.add_argument(
        '--output',
        default=str(EMBEDDINGS_PATH),
        help='Path to save embeddings'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for encoding'
    )
    parser.add_argument(
        '--batched',
        action='store_true',
        help='Use batched encoding (faster if supported)'
    )

    args = parser.parse_args()

    if args.batched:
        generate_embeddings_batched(
            items_path=Path(args.items),
            output_path=Path(args.output),
            batch_size=args.batch_size
        )
    else:
        generate_embeddings(
            items_path=Path(args.items),
            output_path=Path(args.output),
            batch_size=args.batch_size
        )
