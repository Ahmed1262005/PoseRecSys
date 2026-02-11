"""
Engine Factory Module.

Provides factory functions for creating and managing style learning engines.
Engines are created on-demand (lazy loading) to save memory.
"""

from typing import Dict, Optional

from engines.predictive_four_engine import PredictiveFourEngine


# Engine registry - engines are cached by gender
_engines: Dict[str, PredictiveFourEngine] = {}


def normalize_gender(gender: str) -> str:
    """Normalize gender string to 'female' or 'male'."""
    if gender.lower() in ("female", "women", "woman", "f", "w"):
        return "female"
    return "male"


def get_engine(gender: str) -> PredictiveFourEngine:
    """
    Get or create a PredictiveFourEngine for the specified gender.
    
    Engines are cached and reused across requests.
    
    Args:
        gender: Gender string ('female', 'women', 'male', 'men', etc.)
        
    Returns:
        PredictiveFourEngine configured for the specified gender
    """
    gender_key = normalize_gender(gender)
    
    if gender_key not in _engines:
        print(f"[EngineFactory] Creating engine for gender: {gender_key}")
        _engines[gender_key] = PredictiveFourEngine(gender=gender_key)
    
    return _engines[gender_key]


def get_women_engine() -> PredictiveFourEngine:
    """
    Get the women's fashion engine.
    
    Convenience wrapper around get_engine("female").
    
    Returns:
        PredictiveFourEngine configured for women's fashion
    """
    return get_engine("female")


def get_men_engine() -> PredictiveFourEngine:
    """
    Get the men's fashion engine.
    
    Convenience wrapper around get_engine("male").
    
    Returns:
        PredictiveFourEngine configured for men's fashion
    """
    return get_engine("male")


def get_image_url(item_id: str, gender: str) -> str:
    """
    Generate image URL based on gender.
    
    Args:
        item_id: Item identifier
        gender: Gender string
        
    Returns:
        URL path for the item's image
    """
    gender_key = normalize_gender(gender)
    
    if gender_key == "female":
        # Women's images: /women-images/{category}/{subcategory}/{id}.webp
        return f"/women-images/{item_id.replace(' ', '%20')}.webp"
    else:
        # Men's images: /images/{item_id}.webp
        return f"/images/{item_id.replace(' ', '%20')}.webp"


def clear_engines() -> None:
    """Clear all cached engines. Useful for testing."""
    global _engines
    _engines = {}


# Search engine singleton
_search_engine = None


def get_search_engine():
    """
    Get or create WomenSearchEngine singleton.
    
    Uses Supabase pgvector for similarity search.
    
    Returns:
        WomenSearchEngine instance
    """
    global _search_engine
    if _search_engine is None:
        from women_search_engine import WomenSearchEngine
        _search_engine = WomenSearchEngine()
    return _search_engine
