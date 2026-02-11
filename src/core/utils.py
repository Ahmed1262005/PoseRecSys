"""
Core Utility Functions.

Common utilities used across the application.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse
import numpy as np


# =============================================================================
# Image URL Filtering
# =============================================================================

# Allowed image domains — only S3 and internal
ALLOWED_IMAGE_DOMAINS = frozenset({
    "usepose.s3.us-east-1.amazonaws.com",
})


def is_allowed_image_url(url: Optional[str]) -> bool:
    """
    Check if an image URL is from an allowed domain.

    Only S3/internal domains pass. External retailer domains
    (thereformation.com, rihoas.com, etc.) are blocked.

    Args:
        url: Image URL to check.

    Returns:
        True if URL is from an allowed domain (or is empty/None).
    """
    if not url:
        return True  # empty/None is ok — no image to block
    try:
        netloc = urlparse(url).netloc.lower()
        # Allow any amazonaws.com S3 bucket
        if netloc.endswith(".amazonaws.com"):
            return True
        return netloc in ALLOWED_IMAGE_DOMAINS
    except Exception:
        return False


def filter_gallery_images(images: Optional[List[str]]) -> List[str]:
    """
    Filter a list of gallery image URLs, keeping only allowed domains.

    Args:
        images: List of image URLs (may be None).

    Returns:
        Filtered list with only allowed-domain URLs.
    """
    if not images:
        return []
    return [url for url in images if isinstance(url, str) and is_allowed_image_url(url)]


def sanitize_product_images(product: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize all image fields in a product dict.

    - Nullifies primary_image_url / image_url / hero_image_url if non-allowed
    - Filters gallery_images to only allowed domains

    Args:
        product: Product dict (mutated in-place for performance).

    Returns:
        The same dict, with non-allowed image URLs removed.
    """
    # Primary image
    for key in ("primary_image_url", "image_url", "hero_image_url"):
        val = product.get(key)
        if val and not is_allowed_image_url(val):
            product[key] = None

    # Gallery images
    gallery = product.get("gallery_images")
    if gallery and isinstance(gallery, list):
        product["gallery_images"] = filter_gallery_images(gallery)

    return product


def convert_numpy(obj: Any) -> Any:
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Recursively converts numpy arrays, scalars, and nested structures
    to their Python equivalents.
    
    Args:
        obj: Any object that may contain numpy types
        
    Returns:
        Object with numpy types converted to Python natives
        
    Examples:
        >>> convert_numpy(np.int64(42))
        42
        >>> convert_numpy(np.array([1, 2, 3]))
        [1, 2, 3]
        >>> convert_numpy({'a': np.float32(1.5)})
        {'a': 1.5}
    """
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {convert_numpy(k): convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy(v) for v in obj)
    elif isinstance(obj, set):
        return list(convert_numpy(v) for v in obj)
    return obj


def normalize_string_set(items: List[str]) -> Set[str]:
    """
    Normalize a list of strings to a set of lowercase, stripped strings.
    
    Args:
        items: List of strings (may contain None, empty strings)
        
    Returns:
        Set of normalized strings
    """
    return {s.lower().strip() for s in items if s}


def safe_get(obj: Any, *keys: str, default: Any = None) -> Any:
    """
    Safely get nested dictionary values.
    
    Args:
        obj: Dictionary or object
        *keys: Keys to traverse
        default: Default value if key not found
        
    Returns:
        Value at the nested key path, or default
        
    Example:
        >>> safe_get({'a': {'b': 1}}, 'a', 'b')
        1
        >>> safe_get({'a': {}}, 'a', 'b', default=0)
        0
    """
    current = obj
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        elif hasattr(current, key):
            current = getattr(current, key)
        else:
            return default
        if current is None:
            return default
    return current


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def merge_dicts(*dicts: Dict) -> Dict:
    """
    Merge multiple dictionaries, later dicts override earlier ones.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result
