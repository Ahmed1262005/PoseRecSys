"""
Visual Embedding Module using FashionCLIP
Generates embeddings and builds Faiss index for similarity search
"""
import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image
from tqdm import tqdm
import faiss


class FashionEmbeddingGenerator:
    """
    Generate fashion item embeddings using FashionCLIP
    and build Faiss index for fast similarity search
    """

    def __init__(self, model_name: str = 'fashion-clip', device: str = 'cuda'):
        """
        Initialize FashionCLIP model

        Args:
            model_name: Model identifier (default: 'fashion-clip')
            device: Device to use ('cuda' or 'cpu')
        """
        from fashion_clip.fashion_clip import FashionCLIP

        self.model = FashionCLIP(model_name)
        self.device = device
        self.embedding_dim = 512  # FashionCLIP output dimension

        print(f"FashionCLIP model loaded (dim={self.embedding_dim})")

    def encode_image(self, image_path: str) -> np.ndarray:
        """
        Encode a single image to embedding

        Args:
            image_path: Path to image file

        Returns:
            Normalized 512-dim embedding vector
        """
        image = Image.open(image_path).convert('RGB')
        embedding = self.model.encode_images([image], batch_size=1)[0]

        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding.astype('float32')

    def encode_batch(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode multiple images to embeddings

        Args:
            image_paths: List of paths to image files
            batch_size: Batch size for encoding

        Returns:
            Array of normalized embeddings (N, 512)
        """
        images = []
        valid_indices = []

        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert('RGB')
                images.append(img)
                valid_indices.append(i)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue

        if not images:
            return np.zeros((0, self.embedding_dim), dtype='float32')

        embeddings = self.model.encode_images(images, batch_size=batch_size)

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        return embeddings.astype('float32')

    def generate_all_embeddings(
        self,
        image_dir: str,
        item_ids: List[str],
        batch_size: int = 32,
        extensions: List[str] = ['.jpg', '.jpeg', '.png', '.webp']
    ) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """
        Generate embeddings for all items

        Args:
            image_dir: Directory containing images
            item_ids: List of item IDs to process
            batch_size: Batch size for encoding
            extensions: Valid image file extensions

        Returns:
            Tuple of (embeddings dict, valid item IDs list)
        """
        image_path = Path(image_dir)
        embeddings = {}
        valid_item_ids = []

        # Collect valid image paths
        items_to_process = []
        for item_id in item_ids:
            for ext in extensions:
                img_path = image_path / f"{item_id}{ext}"
                if img_path.exists():
                    items_to_process.append((item_id, str(img_path)))
                    break

        print(f"Found {len(items_to_process)} images out of {len(item_ids)} items")

        # Process in batches
        for i in tqdm(range(0, len(items_to_process), batch_size), desc="Generating embeddings"):
            batch = items_to_process[i:i + batch_size]
            batch_ids = [b[0] for b in batch]
            batch_paths = [b[1] for b in batch]

            batch_embeddings = self.encode_batch(batch_paths, batch_size)

            for item_id, emb in zip(batch_ids, batch_embeddings):
                embeddings[item_id] = emb
                valid_item_ids.append(item_id)

        print(f"Generated {len(embeddings)} embeddings")
        return embeddings, valid_item_ids

    def save_embeddings(
        self,
        embeddings: Dict[str, np.ndarray],
        item_ids: List[str],
        output_path: str
    ):
        """
        Save embeddings to pickle file

        Args:
            embeddings: Dictionary mapping item_id to embedding
            item_ids: List of valid item IDs
            output_path: Path to save pickle file
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        data = {
            'embeddings': embeddings,
            'item_ids': item_ids,
            'embedding_dim': self.embedding_dim,
        }

        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Saved embeddings to {output_path}")

    @staticmethod
    def load_embeddings(embeddings_path: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """
        Load embeddings from pickle file

        Args:
            embeddings_path: Path to pickle file

        Returns:
            Tuple of (embeddings dict, item IDs list)
        """
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)

        return data['embeddings'], data['item_ids']

    def build_faiss_index(
        self,
        embeddings: Union[np.ndarray, Dict[str, np.ndarray]],
        item_ids: Optional[List[str]] = None,
        use_gpu: bool = True
    ) -> faiss.Index:
        """
        Build Faiss index for fast similarity search

        Args:
            embeddings: Either numpy array (N, D) or dict mapping item_id to embedding
            item_ids: List of item IDs (required if embeddings is dict)
            use_gpu: Whether to use GPU for indexing

        Returns:
            Faiss index
        """
        # Convert dict to array if needed
        if isinstance(embeddings, dict):
            if item_ids is None:
                item_ids = list(embeddings.keys())
            embedding_matrix = np.array([embeddings[iid] for iid in item_ids]).astype('float32')
        else:
            embedding_matrix = embeddings.astype('float32')

        # L2 normalize for cosine similarity via inner product
        faiss.normalize_L2(embedding_matrix)

        # Create index
        dim = embedding_matrix.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity for normalized vectors

        # Move to GPU if available and requested
        if use_gpu and faiss.get_num_gpus() > 0:
            gpu_res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_res, 0, index)

        # Add vectors
        index.add(embedding_matrix)

        print(f"Built Faiss index with {index.ntotal} vectors (GPU: {use_gpu and faiss.get_num_gpus() > 0})")
        return index

    @staticmethod
    def save_faiss_index(index: faiss.Index, output_path: str):
        """
        Save Faiss index to file

        Args:
            index: Faiss index
            output_path: Path to save index
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        # Move to CPU if on GPU
        if hasattr(index, 'index'):
            index = faiss.index_gpu_to_cpu(index)

        faiss.write_index(index, output_path)
        print(f"Saved Faiss index to {output_path}")

    @staticmethod
    def load_faiss_index(index_path: str, use_gpu: bool = True) -> faiss.Index:
        """
        Load Faiss index from file

        Args:
            index_path: Path to index file
            use_gpu: Whether to move index to GPU

        Returns:
            Faiss index
        """
        index = faiss.read_index(index_path)

        if use_gpu and faiss.get_num_gpus() > 0:
            gpu_res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_res, 0, index)

        return index


class FaissSearcher:
    """
    Fast similarity search using Faiss index
    """

    def __init__(
        self,
        embeddings_path: str,
        index_path: Optional[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize searcher

        Args:
            embeddings_path: Path to embeddings pickle file
            index_path: Path to Faiss index (optional, will build if not provided)
            use_gpu: Whether to use GPU
        """
        # Load embeddings
        self.embeddings, self.item_ids = FashionEmbeddingGenerator.load_embeddings(embeddings_path)
        self.item_to_idx = {iid: idx for idx, iid in enumerate(self.item_ids)}

        # Load or build index
        if index_path and os.path.exists(index_path):
            self.index = FashionEmbeddingGenerator.load_faiss_index(index_path, use_gpu)
        else:
            gen = FashionEmbeddingGenerator.__new__(FashionEmbeddingGenerator)
            gen.embedding_dim = 512
            self.index = gen.build_faiss_index(self.embeddings, self.item_ids, use_gpu)

        print(f"FaissSearcher ready: {len(self.item_ids)} items, index size: {self.index.ntotal}")

    def search(
        self,
        query_item_id: str,
        k: int = 20,
        exclude_query: bool = True
    ) -> List[Dict]:
        """
        Find similar items

        Args:
            query_item_id: Item ID to search for
            k: Number of results
            exclude_query: Whether to exclude query item from results

        Returns:
            List of dicts with 'item_id' and 'score'
        """
        if query_item_id not in self.embeddings:
            return []

        query = self.embeddings[query_item_id].reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)

        # Search (get extra results in case we exclude query)
        search_k = k + 1 if exclude_query else k
        distances, indices = self.index.search(query, search_k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.item_ids):
                continue

            item_id = self.item_ids[idx]

            if exclude_query and item_id == query_item_id:
                continue

            results.append({
                'item_id': item_id,
                'score': float(score)
            })

            if len(results) >= k:
                break

        return results

    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        k: int = 20
    ) -> List[Dict]:
        """
        Find similar items by embedding vector

        Args:
            query_embedding: Query embedding vector
            k: Number of results

        Returns:
            List of dicts with 'item_id' and 'score'
        """
        query = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)

        distances, indices = self.index.search(query, k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.item_ids):
                continue

            results.append({
                'item_id': self.item_ids[idx],
                'score': float(score)
            })

        return results


def generate_polyvore_embeddings(
    data_dir: str = "data/polyvore",
    images_dir: str = "data/polyvore/images",
    output_embeddings: str = "models/polyvore_embeddings.pkl",
    output_index: str = "models/polyvore_faiss_index.bin",
    batch_size: int = 32
):
    """
    Generate embeddings for entire Polyvore dataset

    Args:
        data_dir: Path to polyvore data
        images_dir: Path to images directory
        output_embeddings: Path to save embeddings
        output_index: Path to save Faiss index
        batch_size: Batch size for encoding
    """
    from data_processing import load_item_metadata

    # Load item IDs
    metadata = load_item_metadata(data_dir)
    item_ids = list(metadata.keys())

    print(f"Processing {len(item_ids)} items...")

    # Generate embeddings
    generator = FashionEmbeddingGenerator()
    embeddings, valid_ids = generator.generate_all_embeddings(
        images_dir, item_ids, batch_size=batch_size
    )

    # Save embeddings
    generator.save_embeddings(embeddings, valid_ids, output_embeddings)

    # Build and save index
    index = generator.build_faiss_index(embeddings, valid_ids)
    generator.save_faiss_index(index, output_index)

    print("Done!")
    return embeddings, valid_ids


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        images_dir = sys.argv[2] if len(sys.argv) > 2 else f"{data_dir}/images"
    else:
        data_dir = "data/polyvore"
        images_dir = "data/polyvore/images"

    generate_polyvore_embeddings(
        data_dir=data_dir,
        images_dir=images_dir,
        output_embeddings="models/polyvore_embeddings.pkl",
        output_index="models/polyvore_faiss_index.bin"
    )
