"""
Build Faiss index for Amazon Fashion FashionCLIP embeddings.
This enables fast nearest neighbor search for candidate generation.
"""

import faiss
import numpy as np
import json
from pathlib import Path
import pickle


def build_faiss_index(
    embeddings_path: str = "data/amazon_fashion/processed/amazon_mens_embeddings_array.npy",
    ids_path: str = "data/amazon_fashion/processed/amazon_mens_embeddings_ids.json",
    output_index_path: str = "models/amazon_mens_faiss.index",
    output_ids_path: str = "models/amazon_mens_faiss_ids.npy",
    use_gpu: bool = True
):
    """Build and save Faiss index from FashionCLIP embeddings."""

    print("=" * 60)
    print("Building Faiss Index for Amazon Fashion Embeddings")
    print("=" * 60)

    # Load embeddings
    print(f"\nLoading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path).astype('float32')
    print(f"Embeddings shape: {embeddings.shape}")

    # Load IDs
    print(f"Loading IDs from {ids_path}...")
    with open(ids_path, 'r') as f:
        ids = json.load(f)
    ids_array = np.array(ids)
    print(f"Number of IDs: {len(ids_array)}")

    # Normalize for cosine similarity (inner product on normalized vectors = cosine)
    print("\nNormalizing embeddings for cosine similarity...")
    faiss.normalize_L2(embeddings)

    d = embeddings.shape[1]  # 512 dimensions
    n = embeddings.shape[0]  # ~59K items

    print(f"Building index: {n} vectors of dimension {d}")

    # For ~60K vectors, IndexFlatIP is fast enough and exact
    # For larger catalogs, consider IndexIVFFlat or IndexHNSW
    index = faiss.IndexFlatIP(d)  # Inner product (cosine after L2 norm)

    if use_gpu and faiss.get_num_gpus() > 0:
        print("Moving index to GPU...")
        gpu_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_res, 0, index)

    # Add vectors to index
    print("Adding vectors to index...")
    index.add(embeddings)
    print(f"Index contains {index.ntotal} vectors")

    # Move back to CPU for saving
    if use_gpu and faiss.get_num_gpus() > 0:
        print("Moving index back to CPU for saving...")
        index = faiss.index_gpu_to_cpu(index)

    # Save index
    Path(output_index_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving index to {output_index_path}...")
    faiss.write_index(index, output_index_path)

    # Save IDs
    print(f"Saving IDs to {output_ids_path}...")
    np.save(output_ids_path, ids_array)

    # Test search
    print("\nTesting index with sample query...")
    test_query = embeddings[0:1]  # First item
    D, I = index.search(test_query, 5)
    print(f"Query item: {ids_array[0]}")
    print(f"Top 5 similar items:")
    for i, (dist, idx) in enumerate(zip(D[0], I[0])):
        print(f"  {i+1}. {ids_array[idx]} (similarity: {dist:.4f})")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return index, ids_array


def load_faiss_index(
    index_path: str = "models/amazon_mens_faiss.index",
    ids_path: str = "models/amazon_mens_faiss_ids.npy",
    use_gpu: bool = True
):
    """Load Faiss index and IDs."""
    index = faiss.read_index(index_path)
    ids_array = np.load(ids_path, allow_pickle=True)

    if use_gpu and faiss.get_num_gpus() > 0:
        gpu_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_res, 0, index)

    return index, ids_array


def search_similar(
    query_embedding: np.ndarray,
    index: faiss.Index,
    ids_array: np.ndarray,
    k: int = 100
) -> list:
    """Search for k most similar items."""
    # Normalize query
    query = query_embedding.astype('float32').reshape(1, -1)
    faiss.normalize_L2(query)

    D, I = index.search(query, k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        results.append({
            'item_id': ids_array[idx],
            'similarity': float(dist)
        })

    return results


if __name__ == "__main__":
    import os
    os.chdir("/home/ubuntu/recSys/outfitTransformer")

    build_faiss_index()
