"""
FashionCLIP encoding and nearest-neighbour style classification.

Reuses the singleton ``PinterestStyleExtractor`` for the actual model;
adds helpers for byte/URL ingestion and style label extraction via
K-nearest product attributes.
"""

from __future__ import annotations

import io
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image
from supabase import Client

from core.logging import get_logger
from integrations.pinterest_style import get_pinterest_style_extractor

logger = get_logger(__name__)

# How many product-attribute dimensions we aggregate
_STYLE_ATTR_KEYS: List[str] = [
    "style_tags",
    "pattern",
    "color_family",
    "formality",
    "occasions",
    "silhouette",
    "fit_type",
    "sleeve_type",
    "neckline",
]

# Map each attribute key to the corresponding feed query-param name
_ATTR_TO_FEED_PARAM: Dict[str, str] = {
    "style_tags": "include_style_tags",
    "pattern": "include_patterns",
    "color_family": "include_color_family",
    "formality": "include_formality",
    "occasions": "include_occasions",
    "silhouette": "include_silhouette",
    "fit_type": "include_fit",
    "sleeve_type": "include_sleeves",
    "neckline": "include_neckline",
}


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def encode_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes into a PIL Image and return a 512-dim FashionCLIP embedding."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    extractor = get_pinterest_style_extractor()
    return extractor.encode_image(image)


def encode_from_url(url: str, timeout: int = 15) -> np.ndarray:
    """Fetch an image from *url* and return a 512-dim FashionCLIP embedding."""
    headers = {"User-Agent": "POSECanvas/1.0"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return encode_from_bytes(resp.content)


# ---------------------------------------------------------------------------
# Style classification via nearest-neighbour product attributes
# ---------------------------------------------------------------------------

def classify_style(
    embedding: np.ndarray,
    supabase: Client,
    nearest_k: int = 20,
) -> Tuple[Optional[str], float, Dict[str, Dict[str, float]]]:
    """
    Classify the style of an embedding by aggregating Gemini attributes
    from the K nearest real products.

    Returns:
        (style_label, confidence, attribute_distributions)

        - style_label: dominant style tag (e.g. "Boho") or ``None``
        - confidence: proportion of nearest products with that tag (0-1)
        - attribute_distributions: ``{attr_key: {value: normalised_score}}``
    """
    # 1. Find K nearest products using the image embedding
    embedding_str = "[" + ",".join(f"{float(v):.8f}" for v in embedding) + "]"

    try:
        result = supabase.rpc(
            "match_products_with_hard_filters",
            {
                "query_embedding": embedding_str,
                "match_count": nearest_k,
            },
        ).execute()
    except Exception:
        logger.exception("match_products_with_hard_filters RPC failed")
        return None, 0.0, {}

    product_rows = result.data if result.data else []
    if not product_rows:
        return None, 0.0, {}

    product_ids = [str(r["product_id"]) for r in product_rows]

    # 2. Fetch Gemini attributes for those products
    attrs_rows = _fetch_product_attributes(supabase, product_ids)
    if not attrs_rows:
        return None, 0.0, {}

    # 3. Aggregate attribute distributions
    distributions = _aggregate_attributes(attrs_rows)

    # 4. Pick dominant style label
    style_dist = distributions.get("style_tags", {})
    if style_dist:
        best_tag = max(style_dist, key=style_dist.get)  # type: ignore[arg-type]
        confidence = style_dist[best_tag]
    else:
        best_tag = None
        confidence = 0.0

    return best_tag, confidence, distributions


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_product_attributes(
    supabase: Client,
    product_ids: List[str],
) -> List[Dict[str, Any]]:
    """Batch-fetch Gemini attribute rows for a list of product IDs.

    Note: ``sleeve_type`` and ``neckline`` live inside the ``construction``
    JSON column, not as top-level columns.  We fetch ``construction`` and
    unpack them in :func:`_unpack_construction`.
    """
    if not product_ids:
        return []

    try:
        result = (
            supabase.table("product_attributes")
            .select("sku_id, style_tags, pattern, color_family, formality, "
                    "occasions, silhouette, fit_type, construction")
            .in_("sku_id", product_ids)
            .execute()
        )
        rows = result.data or []
        # Unpack sleeve_type / neckline from the construction JSON
        for row in rows:
            construction = row.pop("construction", None) or {}
            row["sleeve_type"] = construction.get("sleeve_type")
            row["neckline"] = construction.get("neckline")
        return rows
    except Exception:
        logger.exception("Failed to fetch product_attributes")
        return []


def _aggregate_attributes(
    rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """
    Build normalised frequency distributions for each attribute dimension.

    For scalar columns (e.g. ``pattern``), each row contributes one count.
    For array columns (e.g. ``style_tags``, ``occasions``), each element
    in the array contributes one count.

    Returns ``{attr_key: {value: normalised_count}}`` where values sum to 1.
    """
    counters: Dict[str, Counter] = {k: Counter() for k in _STYLE_ATTR_KEYS}
    n = len(rows)
    if n == 0:
        return {}

    for row in rows:
        for key in _STYLE_ATTR_KEYS:
            raw = row.get(key)
            if raw is None:
                continue

            if isinstance(raw, list):
                # Array column (style_tags, occasions, etc.)
                for item in raw:
                    if item and isinstance(item, str):
                        counters[key][item.strip()] += 1
            elif isinstance(raw, str):
                val = raw.strip()
                if val and val.lower() not in ("n/a", "null", "none", ""):
                    counters[key][val] += 1

    # Normalise each counter into a 0-1 distribution
    distributions: Dict[str, Dict[str, float]] = {}
    for key, counter in counters.items():
        total = sum(counter.values())
        if total > 0:
            distributions[key] = {v: c / total for v, c in counter.most_common()}

    return distributions
