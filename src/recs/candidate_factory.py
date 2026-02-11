"""
Candidate Factory Module

Provides conversion utilities between Candidate objects and Dict representations.

This eliminates manual Candidate construction scattered across the codebase,
particularly in women_search_engine.py where dicts need to be converted for
occasion filtering.
"""

from typing import Dict, Any, Optional, List

from recs.models import Candidate


def candidate_from_dict(data: Dict[str, Any]) -> Candidate:
    """
    Convert a Dict to a Candidate object.

    Handles various field name variations found in the codebase:
    - product_id / item_id
    - primary_image_url / image_url
    - occasions, pattern, formality (from product_attributes)

    Args:
        data: Dict with item data (from search results or DB)

    Returns:
        Candidate object
    """
    return Candidate(
        # ID field - handle variations
        item_id=data.get('product_id') or data.get('item_id') or '',

        # Scores
        embedding_score=float(data.get('embedding_score', 0) or data.get('similarity', 0) or 0),
        preference_score=float(data.get('preference_score', 0) or data.get('preference_boost', 0) or 0),
        sasrec_score=float(data.get('sasrec_score', 0) or 0),
        final_score=float(data.get('final_score', 0) or data.get('similarity', 0) or 0),
        is_oov=data.get('is_oov', False),

        # Metadata
        category=data.get('category') or '',
        broad_category=data.get('broad_category') or '',
        article_type=data.get('article_type') or '',
        brand=data.get('brand') or '',
        price=float(data.get('price', 0) or 0),
        colors=data.get('colors') or [],
        materials=data.get('materials') or [],
        fit=data.get('fit'),
        length=data.get('length'),
        sleeve=data.get('sleeve'),
        neckline=data.get('neckline'),
        rise=data.get('rise'),
        style_tags=data.get('style_tags') or [],
        image_url=data.get('image_url') or data.get('primary_image_url') or '',
        gallery_images=data.get('gallery_images') or [],
        name=data.get('name') or '',

        # product_attributes fields (for direct filtering)
        occasions=data.get('occasions') or [],
        pattern=data.get('pattern'),
        formality=data.get('formality'),
        color_family=data.get('color_family'),
        seasons=data.get('seasons') or [],

        # Coverage & body type (from Gemini Vision)
        coverage_level=data.get('coverage_level'),
        skin_exposure=data.get('skin_exposure'),
        coverage_details=data.get('coverage_details') or [],
        model_body_type=data.get('model_body_type'),
        model_size_estimate=data.get('model_size_estimate'),

        # Source tracking
        source=data.get('source', 'search'),
    )


def candidate_to_dict(candidate: Candidate) -> Dict[str, Any]:
    """
    Convert a Candidate object to a Dict.

    Useful for API responses and serialization.

    Args:
        candidate: Candidate object

    Returns:
        Dict representation
    """
    return {
        "product_id": candidate.item_id,
        "item_id": candidate.item_id,

        # Scores
        "embedding_score": candidate.embedding_score,
        "preference_score": candidate.preference_score,
        "sasrec_score": candidate.sasrec_score,
        "final_score": candidate.final_score,
        "similarity": candidate.final_score,
        "is_oov": candidate.is_oov,

        # Metadata
        "category": candidate.category,
        "broad_category": candidate.broad_category,
        "article_type": candidate.article_type,
        "brand": candidate.brand,
        "price": candidate.price,
        "colors": candidate.colors,
        "materials": candidate.materials,
        "fit": candidate.fit,
        "length": candidate.length,
        "sleeve": candidate.sleeve,
        "neckline": candidate.neckline,
        "rise": candidate.rise,
        "style_tags": candidate.style_tags,
        "image_url": candidate.image_url,
        "primary_image_url": candidate.image_url,
        "gallery_images": candidate.gallery_images,
        "name": candidate.name,

        # product_attributes fields
        "occasions": candidate.occasions,
        "pattern": candidate.pattern,
        "formality": candidate.formality,
        "color_family": candidate.color_family,
        "seasons": candidate.seasons,

        # Coverage & body type
        "coverage_level": candidate.coverage_level,
        "skin_exposure": candidate.skin_exposure,
        "coverage_details": candidate.coverage_details,
        "model_body_type": candidate.model_body_type,
        "model_size_estimate": candidate.model_size_estimate,

        # Source tracking
        "source": candidate.source,
    }


def candidates_from_dicts(data_list: List[Dict[str, Any]]) -> List[Candidate]:
    """
    Convert a list of Dicts to Candidate objects.

    Args:
        data_list: List of dicts with item data

    Returns:
        List of Candidate objects
    """
    return [candidate_from_dict(d) for d in data_list]


def candidates_to_dicts(candidates: List[Candidate]) -> List[Dict[str, Any]]:
    """
    Convert a list of Candidate objects to Dicts.

    Args:
        candidates: List of Candidate objects

    Returns:
        List of dict representations
    """
    return [candidate_to_dict(c) for c in candidates]


def merge_candidate_with_dict(candidate: Candidate, extra_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge a Candidate object with additional data into a Dict.

    Useful when you have a Candidate but need to add search-specific fields
    like keyword_match, brand_match, etc.

    Args:
        candidate: Base Candidate object
        extra_data: Additional fields to include

    Returns:
        Merged dict
    """
    result = candidate_to_dict(candidate)
    result.update(extra_data)
    return result
