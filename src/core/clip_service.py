"""Shared FashionCLIP singleton for text and image encoding.

Consolidates the three independent model instances (WomenSearchEngine,
PinterestStyleExtractor, OutfitEngine) into one GPU-resident model.

Usage::

    from core.clip_service import get_clip_service

    clip = get_clip_service()
    vec = clip.encode_text("red floral dress")           # np.ndarray (512,)
    vecs = clip.encode_text_batch(["dress", "jacket"])    # list[np.ndarray]
    vec = clip.encode_image(pil_image)                    # np.ndarray (512,)
    s = clip.encode_text_pgvector("red floral dress")     # "[0.01,0.02,...]"
    ss = clip.encode_texts_pgvector_batch(["a", "b"])     # list[str]
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import List, Optional

import numpy as np
from PIL import Image

from core.logging import get_logger

logger = get_logger(__name__)

_MODEL_NAME = "patrickjohncyh/fashion-clip"
_EMBEDDING_CACHE_SIZE = 1024  # ~2 MB for 512-dim float32


class CLIPService:
    """Thread-safe, lazy-loaded FashionCLIP encoder (text + image)."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device: str = "cpu"
        self._model_lock = threading.Lock()

        # LRU text-embedding cache (OrderedDict for O(1) move-to-end)
        self._text_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._text_cache_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Double-checked locking: only the first caller loads the model."""
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return

            import torch
            from transformers import CLIPModel, CLIPProcessor

            logger.info("CLIPService: loading FashionCLIP model...")
            model = CLIPModel.from_pretrained(_MODEL_NAME)
            processor = CLIPProcessor.from_pretrained(_MODEL_NAME)

            if torch.cuda.is_available():
                model = model.cuda()
                self._device = "cuda"
                logger.info(
                    "CLIPService: FashionCLIP on GPU",
                    device=torch.cuda.get_device_name(0),
                )
            else:
                self._device = "cpu"
                logger.info("CLIPService: FashionCLIP on CPU")

            model.eval()
            # Assign last to avoid partial initialisation visible to other threads
            self._processor = processor
            self._model = model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device

    # ------------------------------------------------------------------
    # Embedding extraction compat (transformers v4 vs v5+)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text_embedding(emb) -> "torch.Tensor":
        """Handle both raw Tensor (v4) and BaseModelOutput (v5+)."""
        import torch

        if isinstance(emb, torch.Tensor):
            return emb
        if hasattr(emb, "pooler_output") and emb.pooler_output is not None:
            return emb.pooler_output
        if hasattr(emb, "text_embeds") and emb.text_embeds is not None:
            return emb.text_embeds
        raise ValueError(
            f"Unexpected return type from get_text_features: {type(emb)}. "
            f"Attrs: {[a for a in dir(emb) if not a.startswith('_')]}"
        )

    @staticmethod
    def _extract_image_embedding(emb) -> "torch.Tensor":
        """Handle both raw Tensor (v4) and BaseModelOutput (v5+)."""
        import torch

        if isinstance(emb, torch.Tensor):
            return emb
        if hasattr(emb, "image_embeds") and emb.image_embeds is not None:
            return emb.image_embeds
        if hasattr(emb, "pooler_output") and emb.pooler_output is not None:
            return emb.pooler_output
        raise ValueError(
            f"Unexpected return type from get_image_features: {type(emb)}. "
            f"Attrs: {[a for a in dir(emb) if not a.startswith('_')]}"
        )

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def encode_text(self, query: str) -> np.ndarray:
        """Encode a single text to a normalised 512-dim vector (cached)."""
        # Fast-path: check cache without loading the model
        with self._text_cache_lock:
            if query in self._text_cache:
                self._text_cache.move_to_end(query)
                return self._text_cache[query]

        import torch

        self._load_model()

        with torch.no_grad():
            inputs = self._processor(
                text=[query],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            if self._device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            emb = self._model.get_text_features(**inputs)
            emb = self._extract_text_embedding(emb)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            result = emb.cpu().numpy().flatten()

        with self._text_cache_lock:
            self._text_cache[query] = result
            while len(self._text_cache) > _EMBEDDING_CACHE_SIZE:
                self._text_cache.popitem(last=False)

        return result

    def encode_text_batch(self, queries: List[str]) -> List[np.ndarray]:
        """Batch-encode texts in a single forward pass (cached per query)."""
        if not queries:
            return []
        if len(queries) == 1:
            return [self.encode_text(queries[0])]

        results: List[Optional[np.ndarray]] = [None] * len(queries)
        uncached_indices: List[int] = []

        with self._text_cache_lock:
            for i, q in enumerate(queries):
                if q in self._text_cache:
                    self._text_cache.move_to_end(q)
                    results[i] = self._text_cache[q]
                else:
                    uncached_indices.append(i)

        if not uncached_indices:
            return results  # type: ignore[return-value]

        import torch

        self._load_model()
        uncached_queries = [queries[i] for i in uncached_indices]

        with torch.no_grad():
            inputs = self._processor(
                text=uncached_queries,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            if self._device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            emb = self._model.get_text_features(**inputs)
            emb = self._extract_text_embedding(emb)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings_np = emb.cpu().numpy()

        with self._text_cache_lock:
            for j, idx in enumerate(uncached_indices):
                vec = embeddings_np[j].flatten()
                results[idx] = vec
                self._text_cache[queries[idx]] = vec
            while len(self._text_cache) > _EMBEDDING_CACHE_SIZE:
                self._text_cache.popitem(last=False)

        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode a PIL Image to a normalised 512-dim vector."""
        import torch

        self._load_model()

        with torch.no_grad():
            inputs = self._processor(images=[image], return_tensors="pt")
            if self._device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            emb = self._model.get_image_features(**inputs)
            emb = self._extract_image_embedding(emb)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.cpu().numpy().flatten()

    # ------------------------------------------------------------------
    # pgvector-string helpers (used by OutfitEngine)
    # ------------------------------------------------------------------

    def encode_text_pgvector(self, text: str) -> str:
        """Encode text and return as pgvector-compatible string ``[0.01,...]``."""
        vec = self.encode_text(text)
        return "[" + ",".join(map(str, vec.astype("float32").tolist())) + "]"

    def encode_texts_pgvector_batch(self, texts: List[str]) -> List[str]:
        """Batch-encode texts to pgvector strings."""
        vecs = self.encode_text_batch(texts)
        return [
            "[" + ",".join(map(str, v.astype("float32").tolist())) + "]"
            for v in vecs
        ]

    # ------------------------------------------------------------------
    # Warmup (for startup pre-loading)
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """Pre-load model and run a dummy inference pass to warm caches."""
        self._load_model()
        _ = self.encode_text("warmup query")
        logger.info("CLIPService: warmup complete")


# ======================================================================
# Module-level singleton
# ======================================================================

_instance: Optional[CLIPService] = None
_instance_lock = threading.Lock()


def get_clip_service() -> CLIPService:
    """Return the process-wide CLIPService singleton."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = CLIPService()
    return _instance
