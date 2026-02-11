"""Pinterest style extraction using FashionCLIP image embeddings."""

from __future__ import annotations

import io
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image

from core.logging import get_logger
from integrations.pinterest_signals import score_pin_intent

logger = get_logger(__name__)


class PinterestStyleExtractor:
    """Compute a taste vector from Pinterest pin images."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._model_lock = threading.Lock()

    def _load_model(self) -> None:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    import torch
                    from transformers import CLIPModel, CLIPProcessor

                    logger.info("Loading FashionCLIP model for Pinterest embeddings...")
                    model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
                    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

                    if torch.cuda.is_available():
                        model = model.cuda()

                    model.eval()
                    self._model = model
                    self._processor = processor
                    logger.info("FashionCLIP model loaded")

    def encode_image(self, image: Image.Image) -> np.ndarray:
        import torch

        self._load_model()
        if self._processor is None or self._model is None:
            raise RuntimeError("FashionCLIP model failed to initialize")

        with torch.no_grad():
            inputs = self._processor(images=[image], return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            emb = self._model.get_image_features(**inputs)
            if isinstance(emb, torch.Tensor):
                pass
            elif hasattr(emb, "image_embeds") and emb.image_embeds is not None:
                emb = emb.image_embeds
            elif hasattr(emb, "pooler_output") and emb.pooler_output is not None:
                emb = emb.pooler_output
            else:
                raise ValueError(
                    f"Unexpected return type from get_image_features: {type(emb)}"
                )
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.cpu().numpy().flatten()

    def compute_taste_vector(
        self,
        image_urls: List[str],
        max_images: int,
        timeout_seconds: int,
    ) -> Tuple[Optional[List[float]], Dict[str, Any]]:
        used = 0
        failed = 0
        embeddings: List[np.ndarray] = []

        for url in image_urls[:max_images]:
            try:
                image = _fetch_image(url, timeout_seconds)
                if image is None:
                    failed += 1
                    continue
                emb = self.encode_image(image)
                embeddings.append(emb)
                used += 1
            except Exception as exc:
                failed += 1
                logger.warning("Failed to embed Pinterest image", url=url, error=str(exc))

        if not embeddings:
            return None, {"images_used": used, "images_failed": failed}

        mean_vector = np.mean(np.stack(embeddings), axis=0)
        norm = np.linalg.norm(mean_vector)
        if norm > 0:
            mean_vector = mean_vector / norm

        return mean_vector.astype("float32").tolist(), {"images_used": used, "images_failed": failed}


def extract_pin_image_urls(pins: List[Dict[str, Any]]) -> List[str]:
    """Best-effort extraction of image URLs from Pinterest pin objects."""
    urls: List[str] = []
    for pin in pins:
        url = _extract_image_url(pin)
        if url:
            urls.append(url)
    return urls


def extract_pin_preview(pin: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact preview payload for a pin."""
    score, signals, merchant_domain = score_pin_intent(pin)
    return {
        "id": pin.get("id"),
        "title": pin.get("title"),
        "link": pin.get("link"),
        "board_id": pin.get("board_id"),
        "board_section_id": pin.get("board_section_id"),
        "image_url": _extract_image_url(pin),
        "merchant_domain": merchant_domain,
        "shopping_intent_score": score,
        "shopping_signals": signals,
    }


def _extract_image_url(pin: Dict[str, Any]) -> Optional[str]:
    media = pin.get("media") or {}
    images = media.get("images")
    if isinstance(images, dict) and images:
        best_url = None
        best_area = -1
        for item in images.values():
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            width = item.get("width") or 0
            height = item.get("height") or 0
            area = 0
            try:
                area = int(width) * int(height)
            except (TypeError, ValueError):
                area = 0
            if url and area >= best_area:
                best_area = area
                best_url = url
        if best_url:
            return best_url

    media_source = pin.get("media_source") or {}
    if isinstance(media_source, dict):
        url = media_source.get("url")
        if url:
            return url

    return pin.get("image_original_url") or pin.get("image_url")


def _fetch_image(url: str, timeout_seconds: int) -> Optional[Image.Image]:
    headers = {"User-Agent": "FashionRecBot/1.0"}
    resp = requests.get(url, headers=headers, timeout=timeout_seconds)
    if resp.status_code >= 400:
        return None
    try:
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return image
    except Exception:
        return None


_style_extractor: Optional[PinterestStyleExtractor] = None


def get_pinterest_style_extractor() -> PinterestStyleExtractor:
    global _style_extractor
    if _style_extractor is None:
        _style_extractor = PinterestStyleExtractor()
    return _style_extractor
