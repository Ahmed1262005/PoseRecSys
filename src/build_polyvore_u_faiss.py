"""
Build Faiss index from FashionCLIP embeddings for candidate generation.

This creates an index for fast ANN (Approximate Nearest Neighbor) search
to find visually similar items based on FashionCLIP embeddings.
"""
import os
import sys
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_faiss_index(embeddings_path: str, output_path: str):
    """
    Build Faiss index from pre-computed FashionCLIP embeddings.

    Args:
        embeddings_path: Path to .npy file with embeddings (N, 512)
        output_path: Path to save the Faiss index
    """
    try:
        import faiss
    except ImportError:
        print("Installing faiss-cpu...")
        os.system("pip install faiss-cpu")
        import faiss

    print("=" * 60)
    print("Building Faiss Index for Polyvore-U")
    print("=" * 60)

    # Load embeddings
    print(f"\nLoading embeddings from: {embeddings_path}")
    embeddings = np.load(embeddings_path)
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")

    # Convert to float32 (required by Faiss)
    embeddings = embeddings.astype('float32')

    # Normalize for cosine similarity
    # After normalization, Inner Product = Cosine Similarity
    print("\nNormalizing embeddings for cosine similarity...")
    faiss.normalize_L2(embeddings)

    # Build index
    dim = embeddings.shape[1]
    print(f"\nBuilding IndexFlatIP (Inner Product) with dim={dim}...")
    start_time = time.time()

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    build_time = time.time() - start_time
    print(f"  Index built in {build_time:.2f}s")
    print(f"  Total vectors: {index.ntotal}")

    # Test search performance
    print("\nTesting search performance...")
    n_queries = 100
    k = 100

    # Random query vectors
    query_indices = np.random.choice(len(embeddings), n_queries, replace=False)
    query_vectors = embeddings[query_indices]

    start_time = time.time()
    distances, indices = index.search(query_vectors, k)
    search_time = time.time() - start_time

    print(f"  {n_queries} queries with k={k}: {search_time*1000:.2f}ms total")
    print(f"  Average per query: {search_time*1000/n_queries:.3f}ms")

    # Save index
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"\nSaving index to: {output_path}")
    faiss.write_index(index, output_path)

    # Verify saved index
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size: {file_size:.2f} MB")

    # Test loading
    print("\nVerifying saved index...")
    loaded_index = faiss.read_index(output_path)
    print(f"  Loaded index has {loaded_index.ntotal} vectors")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return index


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build Faiss index from FashionCLIP embeddings")
    parser.add_argument(
        '--embeddings',
        default='data/polyvore_u/polyvore_u_clip_embeddings.npy',
        help='Path to embeddings .npy file'
    )
    parser.add_argument(
        '--output',
        default='models/polyvore_u_faiss.index',
        help='Path to save Faiss index'
    )
    args = parser.parse_args()

    build_faiss_index(args.embeddings, args.output)


if __name__ == '__main__':
    main()
