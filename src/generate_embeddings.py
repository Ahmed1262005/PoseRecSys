"""
Generate FashionCLIP embeddings for Polyvore dataset
Maps outfit_id/index.jpg images to item tids
"""
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_mapping(mapping_path: str) -> Dict[str, str]:
    """Load image to tid mapping"""
    with open(mapping_path) as f:
        return json.load(f)

def generate_embeddings(
    images_dir: str,
    mapping_path: str,
    output_embeddings: str,
    output_index: str,
    batch_size: int = 64,
    max_items: int = None
):
    """Generate embeddings for Polyvore images"""
    from fashion_clip.fashion_clip import FashionCLIP
    import faiss

    # Load mapping
    print("Loading mapping...")
    mapping = load_mapping(mapping_path)
    print(f"Loaded {len(mapping)} item mappings")

    # Initialize model
    print("Loading FashionCLIP model...")
    model = FashionCLIP('fashion-clip')
    embedding_dim = 512

    # Collect valid image paths
    print("Finding images...")
    images_base = Path(images_dir) / "images"
    items_to_process = []

    for key, tid in tqdm(mapping.items(), desc="Scanning"):
        set_id, idx = key.rsplit('_', 1)
        img_path = images_base / set_id / f"{idx}.jpg"
        if img_path.exists():
            items_to_process.append((tid, str(img_path)))

    print(f"Found {len(items_to_process)} images")

    if max_items:
        items_to_process = items_to_process[:max_items]
        print(f"Processing first {max_items} items")

    # Process in batches
    embeddings = {}
    valid_ids = []

    for i in tqdm(range(0, len(items_to_process), batch_size), desc="Generating embeddings"):
        batch = items_to_process[i:i + batch_size]
        batch_ids = [b[0] for b in batch]
        batch_paths = [b[1] for b in batch]

        # Load images
        images = []
        valid_batch_ids = []
        for tid, path in zip(batch_ids, batch_paths):
            try:
                img = Image.open(path).convert('RGB')
                images.append(img)
                valid_batch_ids.append(tid)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue

        if not images:
            continue

        # Encode
        batch_embeddings = model.encode_images(images, batch_size=len(images))

        # Normalize
        norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
        batch_embeddings = batch_embeddings / norms

        # Store
        for tid, emb in zip(valid_batch_ids, batch_embeddings):
            embeddings[tid] = emb.astype('float32')
            valid_ids.append(tid)

    print(f"Generated {len(embeddings)} embeddings")

    # Save embeddings
    os.makedirs(os.path.dirname(output_embeddings), exist_ok=True)
    data = {
        'embeddings': embeddings,
        'item_ids': valid_ids,
        'embedding_dim': embedding_dim,
    }
    with open(output_embeddings, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved embeddings to {output_embeddings}")

    # Build Faiss index
    print("Building Faiss index...")
    embedding_matrix = np.array([embeddings[tid] for tid in valid_ids]).astype('float32')
    faiss.normalize_L2(embedding_matrix)

    index = faiss.IndexFlatIP(embedding_dim)

    # Use GPU if available
    if faiss.get_num_gpus() > 0:
        gpu_res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
        gpu_index.add(embedding_matrix)
        # Convert back to CPU for saving
        index = faiss.index_gpu_to_cpu(gpu_index)
    else:
        index.add(embedding_matrix)

    faiss.write_index(index, output_index)
    print(f"Saved Faiss index to {output_index} ({index.ntotal} vectors)")

    return embeddings, valid_ids


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', default='data/polyvore/images')
    parser.add_argument('--mapping', default='data/polyvore/image_to_tid_mapping.json')
    parser.add_argument('--output-embeddings', default='models/polyvore_embeddings.pkl')
    parser.add_argument('--output-index', default='models/polyvore_faiss_index.bin')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-items', type=int, default=None)
    args = parser.parse_args()

    generate_embeddings(
        images_dir=args.images_dir,
        mapping_path=args.mapping,
        output_embeddings=args.output_embeddings,
        output_index=args.output_index,
        batch_size=args.batch_size,
        max_items=args.max_items
    )
