#!/usr/bin/env python3
"""
Generate FashionCLIP embeddings for HPImages dataset.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List
import numpy as np
from PIL import Image
from tqdm import tqdm


def generate_hp_embeddings(
    hp_images_dir: str = "/home/ubuntu/recSys/outfitTransformer/HPImages",
    output_path: str = "/home/ubuntu/recSys/outfitTransformer/models/hp_embeddings.pkl",
    batch_size: int = 32
):
    """Generate embeddings for all HP dataset images."""

    # Import FashionCLIP
    from fashion_clip.fashion_clip import FashionCLIP

    print("Loading FashionCLIP model...")
    model = FashionCLIP('fashion-clip')

    hp_dir = Path(hp_images_dir)

    # Find all category folders
    categories = [d for d in hp_dir.iterdir() if d.is_dir()]
    print(f"Found {len(categories)} categories: {[c.name for c in categories]}")

    # Collect all images with metadata
    all_images = []

    for category_dir in categories:
        category_name = category_dir.name
        webp_files = sorted(category_dir.glob("*.webp"), key=lambda x: int(x.stem) if x.stem.isdigit() else 0)

        for img_path in webp_files:
            all_images.append({
                'path': str(img_path),
                'category': category_name,
                'filename': img_path.name,
                'item_id': f"{category_name}/{img_path.stem}"  # e.g., "Plain T-shirts/1"
            })

    print(f"Total images to process: {len(all_images)}")

    # Process in batches
    embeddings_dict = {}
    failed = []

    for i in tqdm(range(0, len(all_images), batch_size), desc="Generating embeddings"):
        batch = all_images[i:i + batch_size]

        # Load images
        images = []
        valid_items = []

        for item in batch:
            try:
                img = Image.open(item['path']).convert('RGB')
                images.append(img)
                valid_items.append(item)
            except Exception as e:
                print(f"Error loading {item['path']}: {e}")
                failed.append(item['item_id'])
                continue

        if not images:
            continue

        # Generate embeddings
        batch_embeddings = model.encode_images(images, batch_size=len(images))

        # L2 normalize
        norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
        batch_embeddings = batch_embeddings / norms
        batch_embeddings = batch_embeddings.astype('float32')

        # Store with metadata
        for item, emb in zip(valid_items, batch_embeddings):
            embeddings_dict[item['item_id']] = {
                'embedding': emb,
                'category': item['category'],
                'filename': item['filename'],
                'path': item['path']
            }

    print(f"\nGenerated {len(embeddings_dict)} embeddings")
    if failed:
        print(f"Failed: {len(failed)} images")

    # Summary by category
    print("\nBy category:")
    category_counts = {}
    for item_id, data in embeddings_dict.items():
        cat = data['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1

    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")

    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(embeddings_dict, f)

    print(f"\nSaved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

    return embeddings_dict


if __name__ == "__main__":
    generate_hp_embeddings()
