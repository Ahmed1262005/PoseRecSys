"""
Local FAISS Vector Store for semantic search.

Replaces Supabase pgvector RPC with in-process FAISS IndexFlatIP for
sub-millisecond vector search. Embeddings are loaded from a disk snapshot
(preferred) or fetched from Supabase on startup (fallback).

Architecture:
    Supabase (source of truth) → snapshot builder → data/faiss/ → LocalVectorStore
                                                                     ↓
    query_embedding → FAISS index.search() → filter in Python → results
                       ~1-5ms               ~0.1ms

Usage:
    store = get_local_vector_store()
    if store.ready:
        results = store.search(query_embedding, limit=100, ...)
"""

import os
import json
import time
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from core.logging import get_logger

logger = get_logger(__name__)

# Default snapshot directory
_DEFAULT_SNAPSHOT_DIR = "data/faiss"


class LocalVectorStore:
    """In-process FAISS vector store for multimodal embeddings.

    Thread-safe singleton. Loads 130K × 512-dim float32 vectors (~250MB)
    into a FAISS IndexFlatIP for exact inner-product search (equivalent
    to cosine similarity on L2-normalized vectors).

    Metadata (brand, price, category, etc.) is stored in a dict for
    fast Python-side post-filtering after FAISS retrieval.
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._index = None                          # faiss.IndexFlatIP
        self._product_ids: List[str] = []           # row i → product_id (str uuid)
        self._metadata: Dict[str, dict] = {}        # product_id → {name, brand, price, ...}
        self._id_to_row: Dict[str, int] = {}        # product_id → FAISS row index
        self._ready = False
        self._count = 0
        self._load_lock = threading.Lock()
        self._initialized = True

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def count(self) -> int:
        return self._count

    # ==================================================================
    # Loading
    # ==================================================================

    def load_snapshot(self, snapshot_dir: str = _DEFAULT_SNAPSHOT_DIR) -> None:
        """Load pre-built FAISS index + metadata from disk.

        Expected files:
            {snapshot_dir}/index.faiss          — FAISS index
            {snapshot_dir}/product_ids.npy      — (N,) string array
            {snapshot_dir}/metadata.pkl         — {product_id: {...}}
            {snapshot_dir}/version.json         — build metadata

        Raises FileNotFoundError if snapshot doesn't exist.
        """
        import faiss

        with self._load_lock:
            p = Path(snapshot_dir)
            index_path = p / "index.faiss"
            ids_path = p / "product_ids.npy"
            meta_path = p / "metadata.pkl"
            version_path = p / "version.json"

            if not index_path.exists():
                raise FileNotFoundError(f"No FAISS snapshot at {index_path}")

            t0 = time.perf_counter()

            self._index = faiss.read_index(str(index_path))
            self._product_ids = list(np.load(str(ids_path), allow_pickle=True))
            with open(meta_path, "rb") as f:
                self._metadata = pickle.load(f)

            self._id_to_row = {pid: i for i, pid in enumerate(self._product_ids)}
            self._count = len(self._product_ids)
            self._ready = True

            elapsed = (time.perf_counter() - t0) * 1000

            # Read version info if available
            version_info = {}
            if version_path.exists():
                with open(version_path) as f:
                    version_info = json.load(f)

            logger.info(
                "FAISS snapshot loaded",
                count=self._count,
                dimension=self._index.d,
                elapsed_ms=round(elapsed, 1),
                built_at=version_info.get("built_at", "unknown"),
            )

    def load_from_supabase(self, supabase_client) -> None:
        """Fallback: download all embeddings from Supabase into FAISS.

        This is slow (~30-60s for 130K vectors) and should only be used
        when no disk snapshot exists. After loading, call save_snapshot()
        to persist for future boots.
        """
        import faiss

        with self._load_lock:
            t0 = time.perf_counter()
            logger.info("Loading embeddings from Supabase (this may take 30-60s)...")

            # 1. Fetch embeddings in batches
            all_embeddings = []
            all_product_ids = []
            batch_size = 1000
            offset = 0

            while True:
                resp = (
                    supabase_client.table("product_multimodal_embeddings")
                    .select("product_id, multimodal_embedding")
                    .eq("version", 1)
                    .range(offset, offset + batch_size - 1)
                    .execute()
                )
                rows = resp.data
                if not rows:
                    break

                for row in rows:
                    pid = str(row["product_id"])
                    emb_raw = row["multimodal_embedding"]
                    # Embedding comes as string "[0.1, 0.2, ...]" from pgvector
                    if isinstance(emb_raw, str):
                        emb = np.array(json.loads(emb_raw), dtype=np.float32)
                    elif isinstance(emb_raw, list):
                        emb = np.array(emb_raw, dtype=np.float32)
                    else:
                        continue
                    all_embeddings.append(emb)
                    all_product_ids.append(pid)

                offset += batch_size
                if len(rows) < batch_size:
                    break

                if offset % 10000 == 0:
                    logger.info(f"  Fetched {offset} embeddings...")

            if not all_embeddings:
                logger.warning("No embeddings found in Supabase")
                return

            # 2. Fetch product metadata
            logger.info(f"Fetched {len(all_embeddings)} embeddings, loading metadata...")
            metadata = {}
            offset = 0
            while True:
                resp = (
                    supabase_client.table("products")
                    .select(
                        "id, name, brand, price, original_price, in_stock, "
                        "primary_image_url, gallery_images, category, "
                        "broad_category, article_type, colors, materials"
                    )
                    .eq("in_stock", True)
                    .range(offset, offset + batch_size - 1)
                    .execute()
                )
                rows = resp.data
                if not rows:
                    break
                for row in rows:
                    pid = str(row["id"])
                    metadata[pid] = {
                        "name": row.get("name", ""),
                        "brand": row.get("brand", ""),
                        "price": float(row.get("price") or 0),
                        "original_price": row.get("original_price"),
                        "in_stock": row.get("in_stock", True),
                        "image_url": row.get("primary_image_url"),
                        "gallery_images": row.get("gallery_images") or [],
                        "category": row.get("category"),
                        "broad_category": row.get("broad_category"),
                        "article_type": row.get("article_type"),
                        "colors": row.get("colors") or [],
                        "materials": row.get("materials") or [],
                    }
                offset += batch_size
                if len(rows) < batch_size:
                    break

            # 3. Fetch Gemini attributes (product_attributes table)
            # These are the fields that _enrich_semantic_results() currently
            # fetches from Algolia at ~2-5s per request. Baking them into
            # the FAISS metadata eliminates that round-trip entirely.
            #
            # Note: neckline, sleeve_type, length live inside the `construction`
            # JSON column, not as top-level columns. We fetch `construction`
            # and extract them during merge.
            logger.info("Loading Gemini attributes from product_attributes...")
            _ATTR_FIELDS = (
                "sku_id, category_l1, category_l2, category_l3, primary_color, "
                "color_family, formality, silhouette, rise, "
                "apparent_fabric, seasons, fit_type, pattern, pattern_scale, "
                "style_tags, occasions, is_fashion_item, "
                "arm_coverage, shoulder_coverage, neckline_depth, "
                "midriff_exposure, back_openness, sheerness_visual, "
                "body_cling_visual, structure_level, drape_level, "
                "cropped_degree, waist_definition_visual, leg_volume_visual, "
                "has_pockets_visible, slit_presence, slit_height, "
                "detail_tags, lining_status_likely, "
                "vibe_tags, coverage_level, styling_role, "
                "construction"
            )
            # Fields to extract from the `construction` JSON column
            _CONSTRUCTION_KEYS = ("neckline", "sleeve_type", "length")
            attr_count = 0
            offset = 0
            while True:
                resp = (
                    supabase_client.table("product_attributes")
                    .select(_ATTR_FIELDS)
                    .range(offset, offset + batch_size - 1)
                    .execute()
                )
                rows = resp.data
                if not rows:
                    break
                for row in rows:
                    pid = str(row.get("sku_id", ""))
                    if pid not in metadata:
                        continue
                    meta = metadata[pid]
                    # Merge all Gemini attributes into metadata
                    for key, val in row.items():
                        if key == "sku_id":
                            continue
                        # Extract nested fields from construction JSON
                        if key == "construction" and isinstance(val, dict):
                            for ck in _CONSTRUCTION_KEYS:
                                cv = val.get(ck)
                                if cv is not None:
                                    meta[ck] = cv
                            continue
                        if val is not None:
                            # Rename has_pockets_visible → has_pockets for compatibility
                            store_key = "has_pockets" if key == "has_pockets_visible" else key
                            meta[store_key] = val
                    attr_count += 1
                offset += batch_size
                if len(rows) < batch_size:
                    break
                if offset % 10000 == 0:
                    logger.info(f"  Fetched {offset} attribute records...")

            logger.info(f"Merged Gemini attributes for {attr_count} products")

            # 3. Build FAISS index
            embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)
            # Normalize for cosine similarity via inner product
            norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings_matrix /= norms

            dim = embeddings_matrix.shape[1]
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(embeddings_matrix)
            self._product_ids = all_product_ids
            self._metadata = metadata
            self._id_to_row = {pid: i for i, pid in enumerate(all_product_ids)}
            self._count = len(all_product_ids)
            self._ready = True

            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                "FAISS index built from Supabase",
                count=self._count,
                dimension=dim,
                metadata_count=len(metadata),
                elapsed_ms=round(elapsed, 1),
            )

    def save_snapshot(self, snapshot_dir: str = _DEFAULT_SNAPSHOT_DIR) -> None:
        """Persist current index + metadata to disk for fast future loads."""
        import faiss

        if not self._ready:
            raise RuntimeError("Cannot save snapshot — store not loaded")

        p = Path(snapshot_dir)
        p.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(p / "index.faiss"))
        np.save(str(p / "product_ids.npy"), np.array(self._product_ids, dtype=object))
        with open(p / "metadata.pkl", "wb") as f:
            pickle.dump(self._metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(p / "version.json", "w") as f:
            json.dump({
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "count": self._count,
                "embedding_version": 1,
                "dimension": self._index.d,
            }, f, indent=2)

        logger.info("FAISS snapshot saved", path=str(p), count=self._count)

    # ==================================================================
    # Search
    # ==================================================================

    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 100,
        exclude_product_ids: Optional[Set[str]] = None,
        include_brands: Optional[List[str]] = None,
        exclude_brands: Optional[List[str]] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
    ) -> List[dict]:
        """Search for similar products using FAISS.

        Over-fetches from FAISS (3-5x limit), applies filters in Python,
        returns top `limit` results. Retries with deeper fetch if filter
        survival is too low.

        Returns list of dicts matching the format expected by
        HybridSearchService._search_multimodal().
        """
        if not self._ready:
            return []

        # Normalize query embedding
        qe = query_embedding.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(qe)
        if norm > 0:
            qe = qe / norm

        # Determine over-fetch factor based on filter complexity
        has_filters = any([
            include_brands, exclude_brands,
            min_price is not None, max_price is not None,
        ])
        overfetch = 5 if has_filters else 3
        faiss_k = min(limit * overfetch, self._count)

        # Normalize filter inputs for case-insensitive matching
        include_brands_lower = (
            {b.lower() for b in include_brands} if include_brands else None
        )
        exclude_brands_lower = (
            {b.lower() for b in exclude_brands} if exclude_brands else None
        )
        exclude_ids = exclude_product_ids or set()

        results = self._fetch_and_filter(
            qe, faiss_k, limit,
            exclude_ids, include_brands_lower, exclude_brands_lower,
            min_price, max_price,
        )

        # Retry with deeper fetch if we didn't get enough
        if len(results) < limit and faiss_k < self._count:
            deeper_k = min(faiss_k * 3, self._count)
            if deeper_k > faiss_k:
                results = self._fetch_and_filter(
                    qe, deeper_k, limit,
                    exclude_ids, include_brands_lower, exclude_brands_lower,
                    min_price, max_price,
                )

        return results[:limit]

    def _fetch_and_filter(
        self,
        query_embedding: np.ndarray,
        faiss_k: int,
        limit: int,
        exclude_ids: Set[str],
        include_brands_lower: Optional[Set[str]],
        exclude_brands_lower: Optional[Set[str]],
        min_price: Optional[float],
        max_price: Optional[float],
    ) -> List[dict]:
        """Run FAISS search + Python post-filter."""
        scores, indices = self._index.search(query_embedding, faiss_k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= self._count:
                continue

            pid = self._product_ids[idx]

            # Exclusion check
            if pid in exclude_ids:
                continue

            meta = self._metadata.get(pid)
            if not meta:
                continue

            # Filter: brand inclusion
            if include_brands_lower:
                if meta.get("brand", "").lower() not in include_brands_lower:
                    continue

            # Filter: brand exclusion
            if exclude_brands_lower:
                if meta.get("brand", "").lower() in exclude_brands_lower:
                    continue

            # Filter: price range
            price = meta.get("price", 0)
            if min_price is not None and price < min_price:
                continue
            if max_price is not None and price > max_price:
                continue

            # Build result dict matching _search_multimodal output format.
            # When Gemini attributes are baked into metadata (v2 snapshot),
            # these fields are populated directly — no Algolia enrichment needed.
            colors = meta.get("colors") or []
            primary_color = meta.get("primary_color") or (colors[0] if colors else None)
            results.append({
                "product_id": pid,
                "name": meta.get("name", ""),
                "brand": meta.get("brand", ""),
                "image_url": meta.get("image_url"),
                "gallery_images": meta.get("gallery_images") or [],
                "price": float(price),
                "original_price": meta.get("original_price"),
                "is_on_sale": meta.get("is_on_sale") if meta.get("is_on_sale") is not None else (
                    meta.get("original_price") is not None
                    and meta.get("original_price", 0) > price
                ),
                # Core Gemini attributes
                "category_l1": meta.get("category_l1"),
                "category_l2": meta.get("category_l2"),
                "category_l3": meta.get("category_l3"),
                "broad_category": meta.get("broad_category") or meta.get("category"),
                "article_type": meta.get("article_type"),
                "primary_color": primary_color,
                "color_family": meta.get("color_family"),
                "pattern": meta.get("pattern"),
                "pattern_scale": meta.get("pattern_scale"),
                "apparent_fabric": meta.get("apparent_fabric"),
                "fit_type": meta.get("fit_type"),
                "formality": meta.get("formality"),
                "silhouette": meta.get("silhouette"),
                "length": meta.get("length"),
                "neckline": meta.get("neckline"),
                "sleeve_type": meta.get("sleeve_type"),
                "rise": meta.get("rise"),
                "style_tags": meta.get("style_tags") or [],
                "occasions": meta.get("occasions") or [],
                "seasons": meta.get("seasons") or [],
                "colors": colors,
                "materials": meta.get("materials") or [],
                "trending_score": meta.get("trending_score"),
                # Coverage attributes (v1.0.0.2)
                "arm_coverage": meta.get("arm_coverage"),
                "shoulder_coverage": meta.get("shoulder_coverage"),
                "neckline_depth": meta.get("neckline_depth"),
                "midriff_exposure": meta.get("midriff_exposure"),
                "back_openness": meta.get("back_openness"),
                "sheerness_visual": meta.get("sheerness_visual"),
                # Shape / Silhouette (v1.0.0.2)
                "body_cling_visual": meta.get("body_cling_visual"),
                "structure_level": meta.get("structure_level"),
                "drape_level": meta.get("drape_level"),
                "cropped_degree": meta.get("cropped_degree"),
                "waist_definition_visual": meta.get("waist_definition_visual"),
                "leg_volume_visual": meta.get("leg_volume_visual"),
                # Details (v1.0.0.2)
                "has_pockets": meta.get("has_pockets"),
                "slit_presence": meta.get("slit_presence"),
                "slit_height": meta.get("slit_height"),
                "detail_tags": meta.get("detail_tags") or [],
                "lining_status_likely": meta.get("lining_status_likely"),
                # Metadata
                "vibe_tags": meta.get("vibe_tags") or [],
                "coverage_level": meta.get("coverage_level"),
                "styling_role": meta.get("styling_role"),
                "semantic_score": float(score),
                "source": "semantic",
            })

            if len(results) >= limit:
                break

        return results

    # ==================================================================
    # Utilities
    # ==================================================================

    def get_embedding(self, product_id: str) -> Optional[np.ndarray]:
        """Retrieve the stored embedding for a product (for debugging)."""
        if not self._ready:
            return None
        row = self._id_to_row.get(product_id)
        if row is None:
            return None
        return self._index.reconstruct(row)


def get_local_vector_store() -> LocalVectorStore:
    """Get the singleton LocalVectorStore instance."""
    return LocalVectorStore()
